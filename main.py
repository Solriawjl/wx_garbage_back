import requests
from fastapi import FastAPI, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from fastapi import File, UploadFile, Form, Depends
import os
from sqlalchemy.sql.expression import func
from sqlalchemy import desc
import random
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware

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

app = FastAPI(title="智能垃圾分类小程序 API", version="1.2")

# ==========================================
# 配置 CORS 跨域
# ==========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8848",
        "http://127.0.0.1:8848",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有请求方法 (GET, POST, PUT, DELETE)
    allow_headers=["*"],  # 允许所有请求头
)

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
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.Hardswish(),
        nn.Dropout(0.5),
        nn.Linear(512, 4) # num_classes = 4
    )
    return model

# 2. 实例化模型并加载权重
ai_model = build_inference_model()
try:
    # 替换为你实际的权重路径
    ai_model.load_state_dict(torch.load("weights/best_mobilenetv3.pth", map_location=device, weights_only=True))
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

# 动态热度计算算法
def calculate_dynamic_heat(created_at, real_click_count: int, article_id: int) -> int:
    """
    动态热度计算函数：基础随机值 + (存活天数 × 日均自然增长) + (真实点击量 × 权重)
    """
    if not created_at:
        created_at = datetime.now()

    # 1. 利用文章 ID 生成固定随机种子，保证同一篇文章的基础热度不会每次刷新都乱跳
    random.seed(article_id)
    base_heat = random.randint(500, 3000)  # 基础热度

    # 2. 计算文章发布了多少天
    days_alive = (datetime.now() - created_at).days
    if days_alive < 0:
        days_alive = 0

    # 3. 时间发酵：每天自然增长 10~50 的热度
    time_bonus = days_alive * random.randint(10, 50)

    # 4. 真实点击的权重放大 (假设真实的 view_count 每次点击+1，我们放大3倍展示)
    real_click_count = real_click_count if real_click_count else 0
    total_heat = base_heat + time_bonus + (real_click_count * 3)

    return total_heat

# ==============================================================================
# 后台管理系统 (Admin Web) 专属 API 组
# ==============================================================================

@app.post("/api/admin/login")
async def admin_login(login_data: schemas.AdminLoginRequest):
    """
    后台管理系统：管理员登录接口
    """
    # 极简验证：写死管理员账号密码
    if login_data.username == "admin" and login_data.password == "123456":
        # 必须返回 Vue 模板期待的数据格式，包含 access_token
        return {
            "code": 200,
            "message": "登录成功",
            "data": {
                "access_token": f"fake-jwt-token-{uuid.uuid4().hex}" # 伪造一个随机 Token
            }
        }
    else:
        return {
            "code": 500,  # 模板通常把非 200 视为错误
            "message": "账号或密码错误，请重试！",
            "data": None
        }

# ==============================================================================
# 登录显示菜单
# ==============================================================================
@app.get("/api/admin/menu/list")
async def get_geeker_menu():
    """
    后台管理系统：动态获取左侧菜单
    """
    return {
        "code": 200,
        "message": "成功",
        "data": [
            {
                "path": "/home/index",
                "name": "home",
                "component": "/home/index",
                "meta": {
                    "icon": "HomeFilled",
                    "title": "首页",
                    "isLink": "",
                    "isHide": False,
                    "isFull": False,
                    "isAffix": True,
                    "isKeepAlive": True
                }
            },
            {
                "path": "/garbage",
                "name": "garbage",
                "redirect": "/garbage/items",
                "meta": {
                    "icon": "List",
                    "title": "垃圾分类管理",
                    "isLink": "",
                    "isHide": False,
                    "isFull": False,
                    "isAffix": False,
                    "isKeepAlive": True
                },
                "children": [
                    {
                        # 这里复用模板的 proTable 页面底子，稍后我们去改造它
                        "path": "/garbage/items",
                        "name": "garbageItems",
                        "component": "/garbage/items/index",
                        "meta": {
                            "icon": "Menu",
                            "title": "物品图鉴词库",
                            "isLink": "",
                            "isHide": False,
                            "isFull": False,
                            "isAffix": False,
                            "isKeepAlive": True
                        }
                    },
                    {
                        "path": "/garbage/categories",
                        "name": "garbageCategories",
                        "component": "/garbage/categories/index", # 暂时指向同一个底子页面
                        "meta": {
                            "icon": "Collection",
                            "title": "四大类科普配置", # 对应你的导师要求的日式严谨教育闭环
                            "isLink": "",
                            "isHide": False,
                            "isFull": False,
                            "isAffix": False,
                            "isKeepAlive": True
                        }
                    }
                ]
            },
            {
                "path": "/feedback",
                "name": "feedbackAudit",
                "component": "/feedback/index",
                "meta": {
                    "icon": "Comment",
                    "title": "用户反馈审核", # 理错题和建议
                    "isLink": "",
                    "isHide": False,
                    "isFull": False,
                    "isAffix": False,
                    "isKeepAlive": True
                }
            },
            {
                "path": "/users",
                "name": "userManage",
                "component": "/users/index",
                "meta": {
                    "icon": "User",
                    "title": "小程序用户管理", # 用户信息
                    "isLink": "",
                    "isHide": False,
                    "isFull": False,
                    "isAffix": False,
                    "isKeepAlive": True
                }
            }
        ]
    }

@app.get("/api/admin/auth/buttons")
async def get_geeker_buttons():
    """
    前端获取按钮权限的接口，直接返回空字典，防止报错即可
    """
    return {
        "code": 200,
        "message": "成功",
        "data": {}
    }

# ==============================================================================
# 后台管理系统 - 垃圾图鉴模块
# ==============================================================================
@app.get("/api/admin/items")
async def get_admin_garbage_items(
    pageNum: int = Query(1, description="当前页码"),
    pageSize: int = Query(10, description="每页数量"),
    item_name: str = Query(None, description="搜索：物品名称"),
    category_type: int = Query(None, description="搜索：所属大类"),
    db: Session = Depends(get_db)
):
    """
    分页获取垃圾物品列表，支持条件筛选
    """
    # 1. 构建查询对象
    query = db.query(models.GarbageItem)

    # 2. 如果前端传了搜索条件，进行过滤
    if item_name:
        query = query.filter(models.GarbageItem.item_name.like(f"%{item_name}%"))
    if category_type:
        query = query.filter(models.GarbageItem.category_type == category_type)

    # 3. 计算总数
    total = query.count()

    # 4. 分页查询
    skip = (pageNum - 1) * pageSize
    items = query.order_by(models.GarbageItem.id.desc()).offset(skip).limit(pageSize).all()

    # 5. 格式化数据返回给 Vue 模板
    list_data = []
    for item in items:
        list_data.append({
            "id": item.id,
            "item_name": item.item_name,
            "category_type": item.category_type,
            "sub_category": item.sub_category,
            "tips": item.tips,
            "image_url": item.image_url,
            "created_at": item.created_at.strftime("%Y-%m-%d %H:%M:%S") if item.created_at else ""
        })

    # 必须严格符合 Geeker-Admin ProTable 的数据结构期望
    return {
        "code": 200,
        "message": "成功",
        "data": {
            "list": list_data,
            "total": total,
            "pageNum": pageNum,
            "pageSize": pageSize
        }
    }


from pydantic import BaseModel
from typing import List, Optional


# --- Admin API 请求体数据模型 ---
class AdminItemSchema(BaseModel):
    item_name: str
    category_type: int
    sub_category: Optional[str] = "其他类"
    tips: Optional[str] = ""
    image_url: Optional[str] = ""


class AdminDeleteSchema(BaseModel):
    id: List[int]


# ==========================================
# 接口：后台新增垃圾物品
# ==========================================
@app.post("/api/admin/items")
async def add_admin_garbage_item(item_data: AdminItemSchema, db: Session = Depends(get_db)):
    new_item = models.GarbageItem(
        item_name=item_data.item_name,
        category_type=item_data.category_type,
        sub_category=item_data.sub_category,
        tips=item_data.tips,
        image_url=item_data.image_url
    )
    db.add(new_item)
    db.commit()
    return {"code": 200, "message": "新增成功", "data": None}


# ==========================================
# 接口：后台修改垃圾物品
# ==========================================
@app.put("/api/admin/items/{item_id}")
async def edit_admin_garbage_item(item_id: int, item_data: AdminItemSchema, db: Session = Depends(get_db)):
    item = db.query(models.GarbageItem).filter(models.GarbageItem.id == item_id).first()
    if not item:
        return {"code": 404, "message": "物品不存在", "data": None}

    item.item_name = item_data.item_name
    item.category_type = item_data.category_type
    item.sub_category = item_data.sub_category
    item.tips = item_data.tips
    item.image_url = item_data.image_url

    db.commit()
    return {"code": 200, "message": "修改成功", "data": None}


# ==========================================
# 接口：后台批量/单条删除垃圾物品
# ==========================================
@app.post("/api/admin/items/delete")
async def delete_admin_garbage_items(req: AdminDeleteSchema, db: Session = Depends(get_db)):
    db.query(models.GarbageItem).filter(models.GarbageItem.id.in_(req.id)).delete(synchronize_session=False)
    db.commit()
    return {"code": 200, "message": "删除成功", "data": None}


# ==============================================================================
# 后台管理系统 - 四大类科普配置模块 (固定4条数据，仅支持修改)
# ==============================================================================
class AdminCategorySchema(BaseModel):
    eco_value: Optional[str] = ""
    put_guidance: Optional[str] = ""
    harm_description: Optional[str] = ""
    process_method: Optional[str] = ""
    sub_guidance: Optional[str] = ""

@app.get("/api/admin/categories")
async def get_admin_categories(db: Session = Depends(get_db)):
    """
    获取四大分类的科普配置列表
    """
    categories = db.query(models.GarbageCategory).order_by(models.GarbageCategory.id.asc()).all()

    list_data = []
    for cat in categories:
        list_data.append({
            "id": cat.id,
            "category_name": cat.category_name,
            "category_class": cat.category_class,
            "eco_value": cat.eco_value,
            "put_guidance": cat.put_guidance,
            "harm_description": cat.harm_description,
            "process_method": cat.process_method,
            "sub_guidance": cat.sub_guidance
        })

    return {
        "code": 200,
        "message": "成功",
        "data": list_data  # 直接返回数组，去掉之前的字典嵌套
    }


@app.put("/api/admin/categories/{category_id}")
async def edit_admin_category(category_id: int, req_data: AdminCategorySchema, db: Session = Depends(get_db)):
    """
    修改特定大类的科普说明
    """
    cat = db.query(models.GarbageCategory).filter(models.GarbageCategory.id == category_id).first()
    if not cat:
        return {"code": 404, "message": "分类不存在"}

    cat.eco_value = req_data.eco_value
    cat.put_guidance = req_data.put_guidance
    cat.harm_description = req_data.harm_description
    cat.process_method = req_data.process_method
    cat.sub_guidance = req_data.sub_guidance

    db.commit()
    return {"code": 200, "message": "科普配置更新成功", "data": None}

class AuditFeedbackSchema(BaseModel):
    id: int
    status: int  # 1-采纳, 2-驳回
    admin_reply: Optional[str] = ""  # 管理员回复字段


# ==========================================
# 接口：后台获取用户反馈列表
# ==========================================
@app.get("/api/admin/feedbacks")
async def get_admin_feedbacks(
        pageNum: int = Query(1),
        pageSize: int = Query(10),
        status: int = Query(None),
        item_name: str = Query(None),
        db: Session = Depends(get_db)
):
    query = db.query(models.Feedback)

    if status is not None:
        query = query.filter(models.Feedback.status == status)
    if item_name:
        query = query.filter(models.Feedback.item_name.like(f"%{item_name}%"))

    total = query.count()
    skip = (pageNum - 1) * pageSize
    feedbacks = query.order_by(models.Feedback.created_at.desc()).offset(skip).limit(pageSize).all()

    list_data = []
    for f in feedbacks:
        list_data.append({
            "id": f.id,
            "user_id": f.user_id,
            "type": f.type.value if hasattr(f.type, 'value') else f.type,
            "image_url": f.image_url,
            "item_name": f.item_name,
            "suggestion": f.suggestion,
            "status": f.status,
            "admin_reply": f.admin_reply if hasattr(f, 'admin_reply') else "",
            "created_at": f.created_at.strftime("%Y-%m-%d %H:%M:%S") if f.created_at else ""
        })

    return {
        "code": 200, "message": "成功",
        "data": {"list": list_data, "total": total, "pageNum": pageNum, "pageSize": pageSize}
    }


# ==========================================
# 接口：后台审核反馈 (采纳/驳回)
# ==========================================
@app.post("/api/admin/feedbacks/audit")
async def audit_admin_feedback(req: AuditFeedbackSchema, db: Session = Depends(get_db)):
    feedback = db.query(models.Feedback).filter(models.Feedback.id == req.id).first()
    if not feedback:
        return {"code": 404, "message": "反馈记录不存在"}

    feedback.status = req.status
    feedback.admin_reply = req.admin_reply

    # 【核心逻辑：真实采纳入库】
    # 如果状态是 1(采纳)，且有图片URL，且是图片类型的反馈
    if req.status == 1 and feedback.image_url and feedback.type in ["image", "图片"]:
        try:
            # 1. 数据清洗：提取纯净的四大类名称
            raw_suggestion = feedback.suggestion if feedback.suggestion else ""
            correct_category = "未分类"
            # 无论后缀带了什么，只要包含了这四个关键字，就精准归类
            for standard_cat in ["可回收物", "有害垃圾", "厨余垃圾", "其他垃圾"]:
                if standard_cat in raw_suggestion:
                    correct_category = standard_cat
                    break

            # 2. 确定保存目录 (现在它只会生成四大类标准文件夹了)
            save_dir = os.path.join("E:/wechat/feedback_image", "train", correct_category)
            os.makedirs(save_dir, exist_ok=True)

            # 3. 下载图片
            response = requests.get(feedback.image_url, timeout=10)
            if response.status_code == 200:
                # 生成唯一文件名
                file_name = f"feedback_{uuid.uuid4().hex[:8]}.jpg"
                file_path = os.path.join(save_dir, file_name)

                # 4. 写入文件
                with open(file_path, "wb") as f:
                    f.write(response.content)
                print(f"✅ 飞轮运转：已成功将纠错照片采纳入训练集 -> {file_path}")
            else:
                print(f"⚠️ 图片下载失败，HTTP状态码: {response.status_code}")

        except Exception as e:
            print(f"❌ 采纳图片处理时发生异常: {e}")

    db.commit()
    return {"code": 200, "message": "审核完成，数据已入库", "data": None}

# ==============================================================================
# 后台管理系统 - 小程序用户管理模块
# ==============================================================================

@app.get("/api/admin/users")
async def get_admin_users(
    pageNum: int = Query(1, description="当前页码"),
    pageSize: int = Query(10, description="每页数量"),
    nickname: str = Query(None, description="搜索：用户昵称"),
    db: Session = Depends(get_db)
):
    """
    分页获取小程序注册用户列表
    """
    query = db.query(models.User)

    # 支持按昵称模糊搜索
    if nickname:
        query = query.filter(models.User.nickname.like(f"%{nickname}%"))

    total = query.count()
    skip = (pageNum - 1) * pageSize
    users = query.order_by(models.User.id.desc()).offset(skip).limit(pageSize).all()

    list_data = []
    for u in users:
        list_data.append({
            "id": u.id,
            "openid": u.openid,  # 微信唯一标识
            "nickname": u.nickname or "微信用户",
            "avatar_url": u.avatar_url or "",
            "score": u.total_score ,  # 积分段位
            "title": u.title if hasattr(u, 'title') and u.title else "环保新手",
            "created_at": u.created_at.strftime("%Y-%m-%d %H:%M:%S") if u.created_at else ""
        })

    return {
        "code": 200,
        "message": "成功",
        "data": {
            "list": list_data,
            "total": total,
            "pageNum": pageNum,
            "pageSize": pageSize
        }
    }

class AdminUserSchema(BaseModel):
    nickname: str
    score: int = 0
    title: Optional[str] = "环保新手"
    avatar_url: Optional[str] = ""  # 接收前端传来的头像URL

class AdminDeleteSchema(BaseModel):
    id: List[int]

@app.post("/api/admin/users")
async def add_admin_user(user_data: AdminUserSchema, db: Session = Depends(get_db)):
    """后台手动新增用户 (主要用于测试或发放虚拟账号)"""
    new_user = models.User(
        openid=f"admin_add_{uuid.uuid4().hex[:8]}",
        nickname=user_data.nickname,
        total_score=user_data.score,
        title=user_data.title,
        avatar_url=user_data.avatar_url # 存入前端传来的头像
    )
    db.add(new_user)
    db.commit()
    return {"code": 200, "message": "新增用户成功", "data": None}

@app.post("/api/admin/users/delete")
async def delete_admin_users(req: AdminDeleteSchema, db: Session = Depends(get_db)):
    """后台删除用户"""
    db.query(models.User).filter(models.User.id.in_(req.id)).delete(synchronize_session=False)
    db.commit()
    return {"code": 200, "message": "删除成功", "data": None}

# web通知
@app.get("/api/admin/notifications")
async def get_notifications(db: Session = Depends(get_db)):
    """获取后台全局通知/待办数量"""
    # 统计状态为 0 (待处理) 的反馈数量
    pending_feedback_count = db.query(models.Feedback).filter(models.Feedback.status == 0).count()

    return {
        "code": 200,
        "message": "成功",
        "data": {
            "pending_feedbacks": pending_feedback_count
        }
    }


# ==============================================================================
# 后台管理系统 - 首页大盘 (Dashboard) 数据统计
# ==============================================================================
from datetime import datetime, timedelta
from sqlalchemy import func
@app.get("/api/admin/dashboard/stats")
async def get_dashboard_stats(db: Session = Depends(get_db)):
    """获取首页大盘统计数据 (完全采用真实数据库数据)"""

    # 1. 基础数据统计
    user_count = db.query(models.User).count()
    item_count = db.query(models.GarbageItem).count()
    pending_feedback = db.query(models.Feedback).filter(models.Feedback.status == 0).count()

    # 真实的累计识别总次数
    recognize_count = db.query(models.RecognizeHistory).count()

    # ==========================================
    # 2. 真实的近 7 天趋势图数据计算
    # ==========================================
    today = datetime.now().date()
    seven_days_ago = today - timedelta(days=6)  # 包含今天在内的过去7天

    # 使用 SQLAlchemy 的 func.date 提取日期，并进行分组统计 (Group By)
    daily_counts = db.query(
        func.date(models.RecognizeHistory.created_at).label("date"),
        func.count(models.RecognizeHistory.id).label("count")
    ).filter(
        func.date(models.RecognizeHistory.created_at) >= seven_days_ago
    ).group_by(
        func.date(models.RecognizeHistory.created_at)
    ).all()

    # 将查询结果转为字典方便查找: { datetime.date(2026, 3, 11): 2, ... }
    count_dict = {row.date: row.count for row in daily_counts}

    # 构建完整的7天X轴和Y轴数据 (关键逻辑：填补那些识别次数为0的日期)
    x_axis = []
    series = []

    for i in range(7):
        current_date = seven_days_ago + timedelta(days=i)
        # 格式化 X 轴显示为 "MM-DD"，例如 "03-11"
        x_axis.append(current_date.strftime("%m-%d"))
        # 从字典中取当天的数据，如果没有就给 0
        series.append(count_dict.get(current_date, 0))

    chart_data = {
        "xAxis": x_axis,
        "series": series
    }

    return {
        "code": 200,
        "message": "成功",
        "data": {
            "user_count": user_count,
            "item_count": item_count,
            "pending_feedback": pending_feedback,
            "recognize_count": recognize_count,
            "chart_data": chart_data
        }
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
    # 根据AI预测结果查询数据库并返回
    # ========================================
    category_info = db.query(models.GarbageCategory).filter(
        models.GarbageCategory.id == predicted_category_id).first()

    if not category_info:
        return {"code": 500, "message": "识别出错，未找到对应分类信息"}

    # 从该大类下随机抽取 4 个具体物品
    suggest_items = db.query(models.GarbageItem).filter(
        models.GarbageItem.category_type == predicted_category_id
    ).order_by(func.rand()).limit(10).all()

    recommend_list = []
    for item in suggest_items:
        recommend_list.append({
            "item_name": item.item_name,
            "tips": item.tips,
            "image_url": item.image_url if item.image_url else "/images/default_item.png"
        })

    # 存入历史记录
    new_history = models.RecognizeHistory(
        user_id=user_id,
        image_url=cos_image_url,
        recognized_name=category_info.category_name,  # 这里用大类名兜底
        category_type=category_info.id,
        confidence=conf_val  # 存入真实的准确率
    )
    db.add(new_history)
    db.commit()

    # 组装给前端的返回结果（加入教育闭环和推荐列表）
    mock_result = {
        "user_id": user_id,
        "image_path": cos_image_url,
        "confidence": conf_val,
        "category_id": category_info.id,
        "category_name": category_info.category_name,
        "category_class": category_info.category_class,

        # 原有基础字段
        "eco_value": category_info.eco_value,
        "put_guidance": category_info.put_guidance,

        # 新增教育闭环与日式严谨标准字段
        "harm_description": category_info.harm_description,
        "process_method": category_info.process_method,
        "sub_guidance": category_info.sub_guidance,

        # 猜你想扔的具体物品列表
        "recommend_items": recommend_list
    }
    # print("准备发给前端的数据：", mock_result)
    return {
        "code": 200,
        "message": "图片上传成功！AI识别完成",
        "data": mock_result
    }

# ==========================================
# 接口：端云协同架构专用 (手机端已完成计算，仅上传结果存历史)
# ==========================================
@app.post("/api/recognize/edge")
async def recognize_garbage_edge(
        user_id: int = Form(..., description="当前用户的ID"),
        predicted_idx: int = Form(..., description="手机端算出来的模型索引(0/1/2/3)"),
        confidence: float = Form(..., description="手机端算出来的置信度"),
        file: UploadFile = File(..., description="用户上传的原图，存入COS备用"),
        db: Session = Depends(get_db)
):
    # 1. 保存图片到腾讯云 COS
    file_ext = file.filename.split(".")[-1]
    new_filename = f"images/edge_temp/{uuid.uuid4().hex}.{file_ext}"
    file_bytes = await file.read()
    cos_image_url = upload_file_to_cos(file_bytes, new_filename)

    if not cos_image_url:
        return {"code": 500, "message": "图片上传云端失败，请稍后重试"}

    # 2. 直接转换手机传来的 ID 并查数据库
    predicted_category_id = MODEL_IDX_TO_DB_ID.get(predicted_idx, 4)
    category_info = db.query(models.GarbageCategory).filter(
        models.GarbageCategory.id == predicted_category_id).first()

    if not category_info:
        return {"code": 500, "message": "分类数据查询异常"}

    # “猜你想扔”
    suggest_items = db.query(models.GarbageItem).filter(
        models.GarbageItem.category_type == predicted_category_id
    ).order_by(func.rand()).limit(10).all()

    recommend_list = []
    for item in suggest_items:
        recommend_list.append({
            "item_name": item.item_name,
            "tips": item.tips,
            "image_url": item.image_url if item.image_url else "/images/default_item.png"
        })

    # 3. 存入历史记录表
    new_history = models.RecognizeHistory(
        user_id=user_id,
        image_url=cos_image_url,
        recognized_name=category_info.category_name,
        category_type=category_info.id,
        confidence=confidence
    )
    db.add(new_history)
    db.commit()

    # 4. 组装结果返回给前端展示（加入教育闭环）
    mock_result = {
        "user_id": user_id,
        "image_path": cos_image_url,
        "confidence": confidence,
        "category_id": category_info.id,
        "category_name": category_info.category_name,
        "category_class": category_info.category_class,

        # 教育闭环与科普字段
        "eco_value": category_info.eco_value,
        "put_guidance": category_info.put_guidance,
        "harm_description": category_info.harm_description,
        "process_method": category_info.process_method,
        "sub_guidance": category_info.sub_guidance,

        # 推荐物品列表
        "recommend_items": recommend_list
    }
    # print("准备发给前端的数据：", mock_result)
    return {
        "code": 200,
        "message": "端云协同处理成功",
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
        "image_url": item.image_url if item.image_url else "/images/null.png",

        # 把教育闭环字段也加入到搜索结果中
        "harm_description": category_info.harm_description,
        "process_method": category_info.process_method,
        "sub_guidance": category_info.sub_guidance
    }

    return {
        "code": 200,
        "message": "查询成功",
        "data": result_data
    }


# ==========================================
# 搜索输入时的实时联想 (Auto-Suggest)
# ==========================================
@app.get("/api/search/suggest")
async def suggest_garbage(
        keyword: str = Query(..., description="用户正在输入的关键词"),
        db: Session = Depends(get_db)
):
    if not keyword.strip():
        return {"code": 200, "data": []}

    # 在数据库中模糊匹配，限制最多返回 10 条结果，防止数据过大
    items = db.query(models.GarbageItem).filter(
        models.GarbageItem.item_name.like(f"%{keyword}%")
    ).limit(10).all()

    # 只提取物品名称，组装成简单的纯文本列表
    suggest_list = [item.item_name for item in items]

    return {
        "code": 200,
        "message": "获取联想词成功",
        "data": suggest_list
    }

# ==========================================
# 动态获取“热门搜索”（从知识库随机推流）
# ==========================================
@app.get("/api/search/hot")
async def get_hot_searches(db: Session = Depends(get_db)):
    """
    每次调用，从 GarbageItem 物品总库中随机抽出 6 个具体的物品名称作为热搜。
    既保证了每次打开页面都有新鲜感，又绝对保证搜出来的词在数据库里有完美的科普结果。
    """
    try:
        # 使用 func.rand() 在 MySQL 中随机排序并取前 6 条
        random_items = db.query(models.GarbageItem).order_by(func.rand()).limit(6).all()
        hot_list = [item.item_name for item in random_items]

        # 完美的兜底机制：万一数据库被清空了，用默认词顶上
        if len(hot_list) < 6:
            default_hot = ['塑料瓶', '废电池', '过期感冒药', '大骨头', '外卖包装', '碎玻璃']
            for item in default_hot:
                if len(hot_list) >= 6:
                    break
                if item not in hot_list:
                    hot_list.append(item)

        return {
            "code": 200,
            "message": "获取热搜成功",
            "data": hot_list
        }
    except Exception as e:
        print(f"获取热搜异常: {e}")
        return {
            "code": 200,
            "message": "兜底热搜",
            "data": ['塑料瓶', '废电池', '过期感冒药', '大骨头', '外卖包装', '碎玻璃']
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

    # 1. 创建一个字典，按 sub_category 进行分组
    grouped_dict = {}
    for item in items:
        sub = item.sub_category or "其他类"
        if sub not in grouped_dict:
            grouped_dict[sub] = []

        grouped_dict[sub].append({
            "id": item.id,
            "item_name": item.item_name,
            "tips": item.tips,
            "image_url": item.image_url if item.image_url else "/images/default_item.png"
        })

    # 2. 将字典转换为前端容易遍历的数组格式
    result_list = []
    for sub_cat, sub_items in grouped_dict.items():
        result_list.append({
            "subCategory": sub_cat,
            "items": sub_items
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
        # 引入动态热度算法
        dynamic_heat = calculate_dynamic_heat(tip.created_at, tip.view_count, tip.id)

        result_list.append({
            "id": tip.id,
            "title": tip.title,
            "content": tip.content,
            "image_url": tip.image_url,
            "view_count": dynamic_heat  # 输出动态计算出的伪真实高热度
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
        # 引入动态热度算法
        dynamic_heat = calculate_dynamic_heat(tip.created_at, tip.view_count, tip.id)

        result_list.append({
            "id": tip.id,
            "title": tip.title,
            "content": tip.content,
            "image_url": tip.image_url,
            "view_count": dynamic_heat,  # 输出动态计算出的伪真实高热度
            "created_at": tip.created_at.strftime("%Y-%m-%d") if tip.created_at else ""
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
# 个人中心专属 API 组
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

# 排行榜接口
@app.get("/api/leaderboard")
def get_leaderboard(db: Session = Depends(get_db)):
    # 查询积分最高的前 10 名用户，按 total_score 降序排列
    top_users = db.query(models.User).order_by(models.User.total_score.desc()).limit(10).all()

    result = []
    for index, user in enumerate(top_users):
        # 组装返回数据，如果没有昵称和头像，就给个默认的兜底
        nickname = getattr(user, 'nickname', None)
        avatar = getattr(user, 'avatar_url', None)

        result.append({
            "rank": index + 1,
            "user_id": user.id,
            "nickname": nickname if nickname else f"环保卫士_{user.id}",
            "avatar_url": avatar if avatar else "https://images-1408449839.cos.ap-chengdu.myqcloud.com/images/user/head.png",
            "total_score": user.total_score,
            "title": getattr(user, 'title', '环保新手')
        })

    return {
        "code": 200,
        "message": "获取成功",
        "data": result
    }

from pydantic import BaseModel
# --- 接收昵称的数据模型 ---
class NicknameUpdate(BaseModel):
    user_id: int
    nickname: str


# 1. 更新用户昵称的接口
@app.post("/api/user/update_nickname")
def update_nickname(request: NicknameUpdate, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.id == request.user_id).first()
    if not user:
        return {"code": 404, "message": "用户不存在"}

    user.nickname = request.nickname
    db.commit()
    return {"code": 200, "message": "昵称更新成功"}


# 2. 更新用户头像的接口 (接收图片文件 -> 上传COS -> 存入数据库)
@app.post("/api/user/update_avatar")
def update_avatar(user_id: int = Form(...), file: UploadFile = File(...), db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        return {"code": 404, "message": "用户不存在"}

    # 读取图片文件字节
    file_bytes = file.file.read()

    # 提取后缀名并生成一个云端唯一文件名 (存放在 avatars 文件夹下)
    ext = os.path.splitext(file.filename)[1]
    if not ext:
        ext = ".jpg"  # 兜底后缀
    cloud_file_name = f"avatars/{uuid.uuid4().hex}{ext}"

    # 调用你已经写好的 COS 上传函数
    cos_url = upload_file_to_cos(file_bytes, cloud_file_name)

    if not cos_url:
        return {"code": 500, "message": "头像上传云端失败"}

    # 将腾讯云返回的公网链接存入数据库
    user.avatar_url = cos_url
    db.commit()

    return {
        "code": 200,
        "message": "头像更新成功",
        "data": {"avatar_url": cos_url}
    }

