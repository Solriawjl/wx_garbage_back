import requests
from fastapi import FastAPI, Depends, HTTPException,Query
from sqlalchemy.orm import Session
from fastapi import File, UploadFile, Form,Depends
import shutil
import os
from sqlalchemy.sql.expression import func
from typing import List
from sqlalchemy import desc

# 导入写好的模块
import models, schemas
from database import engine, get_db
from cos_utils import upload_file_to_cos
import uuid

# 保险，不会重复建表
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="智能垃圾分类小程序 API", version="1.0")

# ==========================================
# 小程序凭证
# ==========================================
WX_APPID = "wxe62fdd0decc4d5b1"
WX_SECRET = "e0ac8e33f4481ec26bd7ce23fe5c379d"

@app.get("/")
def read_root():
    return {"message": "垃圾分类后端服务已成功启动！"}

# ==========================================
# 接口：微信静默登录 / 自动注册
# ==========================================
@app.post("/api/user/login", response_model=schemas.UserResponse)
def wechat_login(request_data: schemas.WxLoginRequest, db: Session = Depends(get_db)):
    """
    接收前端传来的 code，向微信服务器换取 openid，
    并在数据库中进行查找或自动注册。
    """
    # 1. 拼装请求微信服务器的 URL
    url = f"https://api.weixin.qq.com/sns/jscode2session?appid={WX_APPID}&secret={WX_SECRET}&js_code={request_data.code}&grant_type=authorization_code"

    # 2. 发起 GET 请求，拿 code 换取 openid
    response = requests.get(url)
    res_data = response.json()

    # 3. 错误拦截：如果 code 无效或过期，微信会返回 errcode
    if "errcode" in res_data and res_data["errcode"] != 0:
        raise HTTPException(
            status_code=400, 
            detail=f"微信授权失败，错误码：{res_data['errcode']}, 信息：{res_data.get('errmsg')}"
        )

    openid = res_data.get("openid")
    if not openid:
        raise HTTPException(status_code=400, detail="未获取到有效 OpenID")

    # 4. 核心业务逻辑：操作 MySQL 数据库
    # 用 SQLAlchemy 去查查 users 表里有没有这个 openid
    user = db.query(models.User).filter(models.User.openid == openid).first()

    if not user:
        # 如果是第一次来，帮你自动创建一个空账号！
        user = models.User(openid=openid)
        db.add(user)      # 放到暂存区
        db.commit()       # 正式提交保存到 MySQL
        db.refresh(user)  # 刷新一下，为了拿到数据库自动生成的 id 和 created_at

    # 5. 返回用户对象 (FastAPI 会自动用 schemas.UserResponse 把密码等敏感信息过滤掉，转成 JSON 给前端)
    return user


# ==========================================
# 接口：AI 图像识别 (核心 API)
# ==========================================
@app.post("/api/recognize")
async def recognize_garbage(
        user_id: int = Form(..., description="当前用户的ID"),
        file: UploadFile = File(..., description="用户上传的垃圾照片"),
        db: Session = Depends(get_db)  # 注入数据库会话，能查表
):
    """
    接收前端上传的图片 -> 上传腾讯云 COS -> 模拟 AI 识别 -> 从数据库查出分类返回。
    """
    # 1. 生成一个唯一的文件名，防止图片重名覆盖
    file_ext = file.filename.split(".")[-1]
    new_filename = f"images/search_temp/{uuid.uuid4().hex}.{file_ext}"

    # 2. 读取前端传来的图片字节流
    file_bytes = await file.read()

    # 3. 上传到腾讯云 COS！
    cos_image_url = upload_file_to_cos(file_bytes, new_filename)

    if not cos_image_url:
        return {"code": 500, "message": "图片上传云端失败，请稍后重试"}

    # ========================================
    # 4. 模拟 AI 模型的输出结果 (后续这里会替换为真实推理)
    # ========================================
    predicted_category_id = 1
    predicted_confidence = 93.0

    # ========================================
    # 5. 根据 AI 的预测结果，去查数据库
    # ========================================
    category_info = db.query(models.GarbageCategory).filter(models.GarbageCategory.id == predicted_category_id).first()

    if not category_info:
        # 万一查不到（比如预测出了个 5），做个安全拦截
        return {"code": 500, "message": "识别出错，未找到对应分类信息"}

    # ========================================
    # 6. 顺手把这条记录存进个人的历史记录表 (recognize_history)
    # ========================================
    new_history = models.RecognizeHistory(
        user_id=user_id,
        image_url=cos_image_url,  # 存入真实的云端链接！
        recognized_name=category_info.category_name,  # AI识别的名称
        category_type=category_info.id,
        confidence=predicted_confidence
    )
    db.add(new_history)
    db.commit()

    # ========================================
    # 7. 组装最终的数据包返回给前端
    # ========================================
    mock_result = {
        "user_id": user_id,
        "image_path": cos_image_url,  # 以后会换成腾讯云的 url
        "confidence": predicted_confidence,  # 93.0
        "category_id": category_info.id,  # 1
        "category_name": category_info.category_name,  # "可回收物"
        "category_class": category_info.category_class,  # "recycle"
        "eco_value": category_info.eco_value,  # "适宜回收利用和资源化..."
        "put_guidance": category_info.put_guidance  # "轻投轻放；清洁干燥..."
    }

    # 4. 将结果返回给前端
    return {
        "code": 200,
        "message": "图片上传成功！AI识别完成",
        "data": mock_result
    }

# ==========================================
# 接口：文本搜索垃圾分类
# ==========================================
@app.get("/api/search")
async def search_garbage(
    keyword: str = Query(..., description="用户搜索的关键词"),
    db: Session = Depends(get_db)
):
    """
    接收前端传来的关键词，进行模糊搜索。
    """
    # 1. 模糊搜索 garbage_items 表
    # .like(f"%{keyword}%") 表示只要名字里包含这个词就被搜出来
    item = db.query(models.GarbageItem).filter(
        models.GarbageItem.item_name.like(f"%{keyword}%")
    ).first()

    # 如果没查到，返回 404 告诉前端
    if not item:
        return {
            "code": 404,
            "message": f"抱歉，词库暂未收录「{keyword}」，您可以尝试拍照识别或提交反馈。",
            "data": None
        }

    # 2. 如果查到了物品，拿着它的 category_type 去查 garbage_categories 表获取详情
    category_info = db.query(models.GarbageCategory).filter(
        models.GarbageCategory.id == item.category_type
    ).first()

    if not category_info:
        return {"code": 500, "message": "分类数据异常", "data": None}

    # 3. 组装完美的数据包并返回
    result_data = {
        "item_name": item.item_name,             # 数据库里真实的完整名字，比如"苹果核"
        "category_id": category_info.id,         # 3
        "category_name": category_info.category_name,   # "厨余垃圾"
        "category_class": category_info.category_class, # "kitchen"
        "eco_value": category_info.eco_value,           # 科普长文...
        "put_guidance": category_info.put_guidance,     # 投放长文...
        "tips": item.tips,                               # 专属小贴士（如果有的话）
        "image_url": item.image_url if item.image_url else "/images/null.png"
    }

    return {
        "code": 200,
        "message": "查询成功",
        "data": result_data
    }

# ==========================================
# 接口：知识库 - 根据大类获取下属物品列表
# ==========================================
@app.get("/api/knowledge/items")
async def get_knowledge_items(
    category_type: int = Query(..., description="大类ID: 1-可回收, 2-有害, 3-厨余, 4-其他"),
    db: Session = Depends(get_db)
):
    """
    前端知识库页面切换 Tab 时调用，返回该大类下的所有具体物品卡片数据。
    """
    # 1. 去 garbage_items 表里查出所有属于该大类的物品
    items = db.query(models.GarbageItem).filter(
        models.GarbageItem.category_type == category_type
    ).all()

    # 2. 组装成列表
    result_list = []
    for item in items:
        result_list.append({
            "id": item.id,
            "item_name": item.item_name,
            "tips": item.tips,
            # 如果数据库里有图就用图，没图就给个默认小图标
            "image_url": item.image_url if item.image_url else "/images/default_item.png"
        })

    return {
        "code": 200,
        "message": "获取分类物品成功",
        "data": result_list
    }


# ==========================================
# 接口：获取首页科普轮播列表 (随机抽取3条)
# ==========================================
@app.get("/api/tips/carousel")
async def get_tips_carousel(db: Session = Depends(get_db)):
    """
    随机抽取 3 条环保知识，用于首页跑马灯轮播
    """
    # 随机抽取 3 条
    tips = db.query(models.EnvironmentalTip).order_by(func.rand()).limit(3).all()

    result_list = []
    for tip in tips:
        result_list.append({
            "id": tip.id,
            "title": tip.title,
            "content": tip.content,
            "image_url": tip.image_url,
            "view_count": tip.view_count
        })

    return {
        "code": 200,
        "message": "获取成功",
        "data": result_list
    }


# ==========================================
# 接口：获取环保知识列表 (带分页功能)
# ==========================================
@app.get("/api/tips/list")
async def get_tips_list(
        page: int = Query(1, description="第几页"),
        size: int = Query(10, description="每页几条"),
        db: Session = Depends(get_db)
):
    """
    按发布时间倒序，获取科普文章列表
    """
    skip = (page - 1) * size
    # 按时间倒序查出当前页的数据
    tips = db.query(models.EnvironmentalTip).order_by(
        models.EnvironmentalTip.created_at.desc()
    ).offset(skip).limit(size).all()

    # 格式化数据返回
    result_list = []
    for tip in tips:
        result_list.append({
            "id": tip.id,
            "title": tip.title,
            "content": tip.content,
            "image_url": tip.image_url,
            "view_count": tip.view_count,
            "created_at": tip.created_at.strftime("%Y-%m-%d")  # 格式化时间给前端
        })

    return {
        "code": 200,
        "message": "获取成功",
        "data": result_list
    }


# ==========================================
# 接口：随机生成挑战题目 (出题机)
# ==========================================
@app.get("/api/challenge/questions")
async def get_challenge_questions(limit: int = 10, db: Session = Depends(get_db)):
    """
    从 garbage_items 表中随机抽取指定数量（默认10道）的题目
    """
    # 随机抽取 10 个物品
    items = db.query(models.GarbageItem).order_by(func.rand()).limit(limit).all()

    if not items:
        return {"code": 404, "message": "题库太空啦，请先添加一些垃圾数据", "data": []}

    question_list = []
    for item in items:
        # 顺便查出这个物品的正确分类名，方便前端核对
        category = db.query(models.GarbageCategory).filter(models.GarbageCategory.id == item.category_type).first()

        question_list.append({
            "id": item.id,
            "item_name": item.item_name,
            "correct_category_id": item.category_type,
            "correct_category_name": category.category_name if category else "未知",
            "tips": item.tips
        })

    return {
        "code": 200,
        "message": "题目生成成功",
        "data": question_list
    }


# ==========================================
# 接口：提交答题卡 (阅卷机与称号升级核心算法)
# ==========================================
@app.post("/api/challenge/submit")
async def submit_challenge(quiz_data: schemas.QuizSubmitRequest, db: Session = Depends(get_db)):
    """
    接收答题结果，加积分，升称号，存错题，留记录。
    """
    # 1. 查询当前用户
    user = db.query(models.User).filter(models.User.id == quiz_data.user_id).first()
    if not user:
        return {"code": 404, "message": "用户不存在"}

    # 2. 累加积分
    user.total_score += quiz_data.score

    # 3. 🎯 核心算法：动态计算环保称号 (你可以根据需要调整分数线)
    new_title = user.title
    if user.total_score >= 500:
        new_title = "环保宗师"
    elif user.total_score >= 200:
        new_title = "环保卫士"
    elif user.total_score >= 50:
        new_title = "环保达人"
    else:
        new_title = "环保新手"

    user.title = new_title  # 更新称号

    # 4. 插入本次挑战记录 (challenge_history)
    new_history = models.ChallengeHistory(
        user_id=user.id,
        score=quiz_data.score,
        correct_count=quiz_data.correct_count,
        earned_title=new_title
    )
    db.add(new_history)

    # 5. 遍历并插入错题本 (wrong_book)
    for wrong_item in quiz_data.wrong_answers:
        new_wrong = models.WrongBook(
            user_id=user.id,
            item_name=wrong_item.item_name,
            user_answer=wrong_item.user_answer,
            correct_answer=wrong_item.correct_answer
        )
        db.add(new_wrong)

    # 6. 一次性打包提交所有数据库变更 (事务保证，要么全成功要么全失败)
    db.commit()

    return {
        "code": 200,
        "message": "交卷成功！",
        "data": {
            "total_score": user.total_score,
            "current_title": user.title
        }
    }


# ==========================================
# 接口：提交纠错反馈
# ==========================================
@app.post("/api/feedback/submit")
async def submit_feedback(feedback_data: schemas.FeedbackSubmitRequest, db: Session = Depends(get_db)):
    """
    接收用户的纠错反馈，存入数据库。
    """
    # 1. 验证用户是否存在
    user = db.query(models.User).filter(models.User.id == feedback_data.user_id).first()
    if not user:
        return {"code": 404, "message": "用户不存在，请先登录"}

    # 2. 存入数据库
    # 这里会自动把新反馈的 status 设为默认的 0 (待处理)
    new_feedback = models.Feedback(
        user_id=feedback_data.user_id,
        type=feedback_data.type,
        image_url=feedback_data.image_url,
        item_name=feedback_data.item_name,
        suggestion=feedback_data.suggestion
    )

    db.add(new_feedback)
    db.commit()

    return {
        "code": 200,
        "message": "感谢您的反馈，提交成功！",
        "data": None
    }

# ==============================================================================
# 🌟 个人中心专属 API 组
# ==============================================================================

# 1. 获取个人中心首页概览数据
@app.get("/api/user/info")
async def get_user_info(user_id: int, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        return {"code": 404, "message": "用户不存在", "data": None}

    # 动态统计识别次数和通关次数
    recognize_count = db.query(models.RecognizeHistory).filter(models.RecognizeHistory.user_id == user_id).count()
    challenge_count = db.query(models.ChallengeHistory).filter(models.ChallengeHistory.user_id == user_id).count()

    return {
        "code": 200,
        "data": {
            "total_score": user.total_score,
            "title": user.title,
            "recognize_count": recognize_count,
            "challenge_count": challenge_count
        }
    }


# 2. 获取识别历史列表
@app.get("/api/user/recognize_history")
async def get_recognize_history(user_id: int, db: Session = Depends(get_db)):
    histories = db.query(models.RecognizeHistory).filter(
        models.RecognizeHistory.user_id == user_id
    ).order_by(desc(models.RecognizeHistory.created_at)).all()

    result = []
    for h in histories:
        # 联表查分类信息获取颜色和中文名
        cat = db.query(models.GarbageCategory).filter(models.GarbageCategory.id == h.category_type).first()
        result.append({
            "id": str(h.id),
            "itemName": h.recognized_name,
            "categoryName": cat.category_name if cat else "未知",
            "categoryClass": cat.category_class if cat else "other",
            "imageUrl": h.image_url,
            "date": h.created_at.strftime("%Y-%m-%d %H:%M") if h.created_at else "",
            "confidence": h.confidence
        })
    return {"code": 200, "data": result}


# 3. 获取挑战历史列表
@app.get("/api/user/challenge_history")
async def get_challenge_history(user_id: int, db: Session = Depends(get_db)):
    histories = db.query(models.ChallengeHistory).filter(
        models.ChallengeHistory.user_id == user_id
    ).order_by(desc(models.ChallengeHistory.created_at)).all()

    result = []
    for h in histories:
        # 动态计算颜色阶级
        t_class = 'level-1'
        if h.earned_title in ['环保王者', '环保宗师']:
            t_class = 'level-4'
        elif h.earned_title == '环保达人':
            t_class = 'level-3'
        elif h.earned_title == '环保卫士':
            t_class = 'level-2'

        result.append({
            "id": str(h.id),
            "score": h.score,
            "correctCount": h.correct_count,
            "title": h.earned_title,
            "titleClass": t_class,
            "date": h.created_at.strftime("%Y-%m-%d %H:%M") if h.created_at else "",
            "wrongList": []  # 此处省略了历史单次错题明细，直接让前端展示分数
        })
    return {"code": 200, "data": result}


# 4. 获取我的错题本
@app.get("/api/user/wrong_book")
async def get_wrong_book(user_id: int, db: Session = Depends(get_db)):
    wrongs = db.query(models.WrongBook).filter(
        models.WrongBook.user_id == user_id
    ).order_by(desc(models.WrongBook.created_at)).all()

    result = []
    for w in wrongs:
        result.append({
            "id": str(w.id),
            "name": w.item_name,
            "userSelect": w.user_answer,
            "correctAnswer": w.correct_answer
        })
    return {"code": 200, "data": result}


# 5. 删除单道错题
@app.delete("/api/user/wrong_book/{item_id}")
async def delete_wrong_item(item_id: int, db: Session = Depends(get_db)):
    item = db.query(models.WrongBook).filter(models.WrongBook.id == item_id).first()
    if item:
        db.delete(item)
        db.commit()
    return {"code": 200, "message": "删除成功"}


# 6. 一键清空错题本
@app.delete("/api/user/wrong_book/clear")
async def clear_wrong_book(user_id: int, db: Session = Depends(get_db)):
    db.query(models.WrongBook).filter(models.WrongBook.user_id == user_id).delete()
    db.commit()
    return {"code": 200, "message": "清空成功"}


# 7. 获取反馈历史列表
@app.get("/api/user/feedback_history")
async def get_feedback_history(user_id: int, db: Session = Depends(get_db)):
    feedbacks = db.query(models.Feedback).filter(
        models.Feedback.user_id == user_id
    ).order_by(desc(models.Feedback.created_at)).all()

    result = []
    for f in feedbacks:
        # 注意：因为你的 type 是 Enum 类型，如果是 Python 的 Enum，需要用 f.type.value 提取字符串
        # 如果你定义的是 SQLAlchemy 的字符串 Enum，直接取值即可。
        type_str = f.type.value if hasattr(f.type, 'value') else f.type

        result.append({
            "id": str(f.id),
            "type": type_str,  # 返回给前端 'image' 或 'text'
            "imageUrl": f.image_url,
            "itemName": f.item_name,
            "suggestion": f.suggestion,
            "status": f.status,
            "adminReply": f.admin_reply,
            "date": f.created_at.strftime("%Y-%m-%d %H:%M") if f.created_at else ""
        })
    return {"code": 200, "data": result}