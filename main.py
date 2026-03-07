import requests
from fastapi import FastAPI, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from fastapi import File, UploadFile, Form, Depends
import shutil
import os
from sqlalchemy.sql.expression import func
from typing import List
from sqlalchemy import desc

# 模型
import io
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as transforms
from PIL import Image

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

# ==========================================
# 全局加载 AI 模型 (以 MobileNetV3 为例)
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"后端 AI 模型推理正在使用设备: {device}")

# 1. 网络结构
def build_inference_model():
    model = tv_models.mobilenet_v3_large(weights=None) # 推理时不需要下预训练权重
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 1024),
        nn.BatchNorm1d(1024),
        nn.Hardswish(),
        nn.Dropout(0.3),
        nn.Linear(1024, 4) # num_classes = 4
    )
    return model

# 2. 实例化模型并加载权重
ai_model = build_inference_model()
try:
    # 替换为你实际的权重路径
    ai_model.load_state_dict(torch.load("weights/best_mobilenetv3.pth", map_location=device))
    ai_model.to(device)
    ai_model.eval() # 切换到评估模式，关闭 Dropout 和 BatchNorm 的动态更新
    print("AI 模型权重加载成功！")
except Exception as e:
    print(f"模型权重加载失败，请检查路径: {e}")

# 3. 验证集预处理参数
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 4. 定义【模型索引】到【数据库分类 ID】的映射字典
# 模型: 0-Kitchen(厨余), 1-Recyclable(可回收), 2-Hazardous(有害), 3-Other(其他)
# 数据库ID: 1-可回收, 2-有害, 3-厨余, 4-其他
MODEL_IDX_TO_DB_ID = {
    0: 3, # 模型出 0 -> 对应数据库 3 (厨余)
    1: 1, # 模型出 1 -> 对应数据库 1 (可回收)
    2: 2, # 模型出 2 -> 对应数据库 2 (有害)
    3: 4  # 模型出 3 -> 对应数据库 4 (其他)
}

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
    url = f"https://api.weixin.qq.com/sns/jscode2session?appid={WX_APPID}&secret={WX_SECRET}&js_code={request_data.code}&grant_type=authorization_code"

    response = requests.get(url)
    res_data = response.json()

    if "errcode" in res_data and res_data["errcode"] != 0:
        raise HTTPException(
            status_code=400,
            detail=f"微信授权失败，错误码：{res_data['errcode']}, 信息：{res_data.get('errmsg')}"
        )

    openid = res_data.get("openid")
    if not openid:
        raise HTTPException(status_code=400, detail="未获取到有效 OpenID")

    user = db.query(models.User).filter(models.User.openid == openid).first()

    if not user:
        user = models.User(openid=openid)
        db.add(user)
        db.commit()
        db.refresh(user)

    return user


# ==========================================
# 接口：AI 图像识别 (真实 AI 推理版)
# ==========================================
@app.post("/api/recognize")
async def recognize_garbage(
        user_id: int = Form(..., description="当前用户的ID"),
        file: UploadFile = File(..., description="用户上传的垃圾照片"),
        db: Session = Depends(get_db)
):
    # 1. 保存图片到腾讯云 (保持原有逻辑)
    file_ext = file.filename.split(".")[-1]
    new_filename = f"images/search_temp/{uuid.uuid4().hex}.{file_ext}"
    file_bytes = await file.read()
    cos_image_url = upload_file_to_cos(file_bytes, new_filename)

    if not cos_image_url:
        return {"code": 500, "message": "图片上传云端失败，请稍后重试"}

    # ========================================
    # AI视觉推理阶段
    # ========================================
    try:
        # A. 将前端传来的字节流转为 PIL 图像
        image = Image.open(io.BytesIO(file_bytes))
        # 防呆处理：兼容 RGBA 或灰度图
        if image.mode != "RGB":
            image = image.convert("RGB")

        # B. 预处理
        input_tensor = image_transforms(image).unsqueeze(0).to(device)  # 增加 batch 维度

        # C. 模型前向传播
        with torch.no_grad():
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = ai_model(input_tensor)
                # 使用 softmax 计算各类别概率
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

                # 获取最大概率的索引和置信度
                confidence, predicted_idx = torch.max(probabilities, 0)

                pred_idx_val = predicted_idx.item()
                conf_val = round(confidence.item() * 100, 2)  # 转为百分比保留2位小数

        # D. 将模型预测的索引转换为数据库的真实分类 ID
        predicted_category_id = MODEL_IDX_TO_DB_ID.get(pred_idx_val, 4)  # 兜底分为其他垃圾

        print(
            f"--> AI 推理完毕 | 模型原始索引: {pred_idx_val} | 映射数据库ID: {predicted_category_id} | 置信度: {conf_val}%")

    except Exception as e:
        print(f"推理时发生错误: {e}")
        return {"code": 500, "message": "AI 模型推理失败，请检查图像格式"}

    # ========================================
    # 根据AI预测结果查询数据库并返回 (保持原有逻辑)
    # ========================================
    category_info = db.query(models.GarbageCategory).filter(models.GarbageCategory.id == predicted_category_id).first()

    if not category_info:
        return {"code": 500, "message": "识别出错，未找到对应分类信息"}

    new_history = models.RecognizeHistory(
        user_id=user_id,
        image_url=cos_image_url,
        recognized_name=category_info.category_name,  # 这里用大类名兜底
        category_type=category_info.id,
        confidence=conf_val  # 存入真实的准确率
    )
    db.add(new_history)
    db.commit()

    mock_result = {
        "user_id": user_id,
        "image_path": cos_image_url,
        "confidence": conf_val,  # 返回真实准确率
        "category_id": category_info.id,
        "category_name": category_info.category_name,
        "category_class": category_info.category_class,
        "eco_value": category_info.eco_value,
        "put_guidance": category_info.put_guidance
    }

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
    item = db.query(models.GarbageItem).filter(
        models.GarbageItem.item_name.like(f"%{keyword}%")
    ).first()

    if not item:
        return {
            "code": 404,
            "message": f"抱歉，词库暂未收录「{keyword}」，您可以尝试拍照识别或提交反馈。",
            "data": None
        }

    category_info = db.query(models.GarbageCategory).filter(
        models.GarbageCategory.id == item.category_type
    ).first()

    if not category_info:
        return {"code": 500, "message": "分类数据异常", "data": None}

    result_data = {
        "item_name": item.item_name,
        "category_id": category_info.id,
        "category_name": category_info.category_name,
        "category_class": category_info.category_class,
        "eco_value": category_info.eco_value,
        "put_guidance": category_info.put_guidance,
        "tips": item.tips,
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
    items = db.query(models.GarbageItem).filter(
        models.GarbageItem.category_type == category_type
    ).all()

    result_list = []
    for item in items:
        result_list.append({
            "id": item.id,
            "item_name": item.item_name,
            "tips": item.tips,
            "image_url": item.image_url if item.image_url else "/images/default_item.png"
        })

    return {
        "code": 200,
        "message": "获取分类物品成功",
        "data": result_list
    }


# ==========================================
# 接口：获取首页科普轮播列表
# ==========================================
@app.get("/api/tips/carousel")
async def get_tips_carousel(db: Session = Depends(get_db)):
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
# 接口：获取环保知识列表
# ==========================================
@app.get("/api/tips/list")
async def get_tips_list(
        page: int = Query(1, description="第几页"),
        size: int = Query(10, description="每页几条"),
        db: Session = Depends(get_db)
):
    skip = (page - 1) * size
    tips = db.query(models.EnvironmentalTip).order_by(
        models.EnvironmentalTip.created_at.desc()
    ).offset(skip).limit(size).all()

    result_list = []
    for tip in tips:
        result_list.append({
            "id": tip.id,
            "title": tip.title,
            "content": tip.content,
            "image_url": tip.image_url,
            "view_count": tip.view_count,
            "created_at": tip.created_at.strftime("%Y-%m-%d")
        })

    return {
        "code": 200,
        "message": "获取成功",
        "data": result_list
    }


# ==========================================
# 接口：随机生成挑战题目
# ==========================================
@app.get("/api/challenge/questions")
async def get_challenge_questions(limit: int = 10, db: Session = Depends(get_db)):
    items = db.query(models.GarbageItem).order_by(func.rand()).limit(limit).all()

    if not items:
        return {"code": 404, "message": "题库太空啦，请先添加一些垃圾数据", "data": []}

    question_list = []
    for item in items:
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
# 接口：提交答题卡
# ==========================================
@app.post("/api/challenge/submit")
async def submit_challenge(quiz_data: schemas.QuizSubmitRequest, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.id == quiz_data.user_id).first()
    if not user:
        return {"code": 404, "message": "用户不存在"}

    user.total_score += quiz_data.score

    new_title = user.title
    if user.total_score >= 500:
        new_title = "环保宗师"
    elif user.total_score >= 200:
        new_title = "环保卫士"
    elif user.total_score >= 50:
        new_title = "环保达人"
    else:
        new_title = "环保新手"

    user.title = new_title

    new_history = models.ChallengeHistory(
        user_id=user.id,
        score=quiz_data.score,
        correct_count=quiz_data.correct_count,
        earned_title=new_title
    )
    db.add(new_history)

    for wrong_item in quiz_data.wrong_answers:
        new_wrong = models.WrongBook(
            user_id=user.id,
            item_name=wrong_item.item_name,
            user_answer=wrong_item.user_answer,
            correct_answer=wrong_item.correct_answer
        )
        db.add(new_wrong)

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
    user = db.query(models.User).filter(models.User.id == feedback_data.user_id).first()
    if not user:
        return {"code": 404, "message": "用户不存在，请先登录"}

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
            "wrongList": []
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


# 7. 获取反馈历史列表
@app.get("/api/user/feedback_history")
async def get_feedback_history(user_id: int, db: Session = Depends(get_db)):
    feedbacks = db.query(models.Feedback).filter(
        models.Feedback.user_id == user_id
    ).order_by(desc(models.Feedback.created_at)).all()

    result = []
    for f in feedbacks:
        type_str = f.type.value if hasattr(f.type, 'value') else f.type

        result.append({
            "id": str(f.id),
            "type": type_str,
            "imageUrl": f.image_url,
            "itemName": f.item_name,
            "suggestion": f.suggestion,
            "status": f.status,
            "adminReply": f.admin_reply,
            "date": f.created_at.strftime("%Y-%m-%d %H:%M") if f.created_at else ""
        })
    return {"code": 200, "data": result}

# ==============================================================================
# 所有清空 (clear) 接口必须定义在单条删除 ({item_id}) 接口上方
# ==============================================================================

# --- 错题本 删除接口 ---
@app.delete("/api/user/wrong_book/clear")
async def clear_wrong_book(user_id: int, db: Session = Depends(get_db)):
    db.query(models.WrongBook).filter(models.WrongBook.user_id == user_id).delete()
    db.commit()
    return {"code": 200, "message": "清空成功"}

@app.delete("/api/user/wrong_book/{item_id}")
async def delete_wrong_item(item_id: int, db: Session = Depends(get_db)):
    item = db.query(models.WrongBook).filter(models.WrongBook.id == item_id).first()
    if item:
        db.delete(item)
        db.commit()
    return {"code": 200, "message": "删除成功"}

# --- 识别历史 删除接口 ---
@app.delete("/api/user/recognize_history/clear")
async def clear_recognize_history(user_id: int, db: Session = Depends(get_db)):
    db.query(models.RecognizeHistory).filter(models.RecognizeHistory.user_id == user_id).delete()
    db.commit()
    return {"code": 200, "message": "清空成功"}

@app.delete("/api/user/recognize_history/{item_id}")
async def delete_recognize_history(item_id: int, db: Session = Depends(get_db)):
    db.query(models.RecognizeHistory).filter(models.RecognizeHistory.id == item_id).delete()
    db.commit()
    return {"code": 200, "message": "删除成功"}

# --- 挑战历史 删除接口 (带扣分与掉段逻辑) ---
@app.delete("/api/user/challenge_history/clear")
async def clear_challenge_history(user_id: int, db: Session = Depends(get_db)):
    # 1. 查出该用户所有的挑战记录
    histories = db.query(models.ChallengeHistory).filter(models.ChallengeHistory.user_id == user_id).all()
    if not histories:
        return {"code": 200, "message": "已清空"}

    # 2. 计算要扣除的总分数
    total_deduct = sum([h.score for h in histories])

    # 3. 一键清空所有记录
    db.query(models.ChallengeHistory).filter(models.ChallengeHistory.user_id == user_id).delete()

    # 4. 联动扣分与掉段
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if user:
        user.total_score = max(0, user.total_score - total_deduct)

        new_title = "环保新手"
        if user.total_score >= 100:
            new_title = "环保王者"
        elif user.total_score >= 50:
            new_title = "环保达人"
        elif user.total_score >= 20:
            new_title = "环保卫士"

        user.title = new_title

    db.commit()
    return {
        "code": 200,
        "message": "清空成功",
        "data": {
            "total_score": user.total_score if user else 0,
            "title": user.title if user else "环保新手"
        }
    }

@app.delete("/api/user/challenge_history/{item_id}")
async def delete_challenge_history(item_id: int, db: Session = Depends(get_db)):
    # 1. 先查出这条记录，获取要扣除的分数和用户ID
    history = db.query(models.ChallengeHistory).filter(models.ChallengeHistory.id == item_id).first()
    if not history:
        return {"code": 404, "message": "记录不存在"}

    user_id = history.user_id
    score_to_deduct = history.score

    # 2. 删除记录
    db.delete(history)

    # 3. 联动扣除用户积分，并重新计算段位称号
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if user:
        # 防止积分扣成负数
        user.total_score = max(0, user.total_score - score_to_deduct)

        # 重新评定称号
        new_title = "环保新手"
        if user.total_score >= 100:
            new_title = "环保王者"
        elif user.total_score >= 50:
            new_title = "环保达人"
        elif user.total_score >= 20:
            new_title = "环保卫士"

        user.title = new_title

    db.commit()

    # 4. 把最新的积分和称号返回给前端，用于刷新缓存
    return {
        "code": 200,
        "message": "删除成功",
        "data": {
            "total_score": user.total_score if user else 0,
            "title": user.title if user else "环保新手"
        }
    }

# --- 反馈历史 删除接口 ---
@app.delete("/api/user/feedback_history/clear")
async def clear_feedback_history(user_id: int, db: Session = Depends(get_db)):
    db.query(models.Feedback).filter(models.Feedback.user_id == user_id).delete()
    db.commit()
    return {"code": 200, "message": "清空成功"}

@app.delete("/api/user/feedback_history/{item_id}")
async def delete_feedback_history(item_id: int, db: Session = Depends(get_db)):
    db.query(models.Feedback).filter(models.Feedback.id == item_id).delete()
    db.commit()
    return {"code": 200, "message": "删除成功"}