import requests
from fastapi import FastAPI, Depends, HTTPException, Query, File, UploadFile, Form
from sqlalchemy.orm import Session
from sqlalchemy.sql.expression import func
from sqlalchemy import desc
import os
import random
import io
import uuid
from datetime import datetime, date, timedelta
from typing import List
from contextlib import asynccontextmanager
from pydantic import BaseModel

# AI 框架
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as transforms
from PIL import Image

# 定时任务
from apscheduler.schedulers.background import BackgroundScheduler

# 导入写好的模块
import models, schemas
from database import engine, get_db, SessionLocal
from cos_utils import upload_file_to_cos
from fastapi.middleware.cors import CORSMiddleware

# 保险，会自动根据模型建表
models.Base.metadata.create_all(bind=engine)


# ==========================================
# 👑 核心逻辑：每周排行榜结算函数 (含通知发放)
# ==========================================
def run_weekly_settlement():
    # 因为定时任务不在 FastAPI 的 HTTP 请求上下文中，需要手动开启独立数据库会话
    db = SessionLocal()
    try:
        print("⏰ 开始执行：每周排行榜结算自动化任务...")

        # 1. 获取全班排名前 3 的学生（按环保星 total_score 降序）
        top_students = db.query(models.User).filter(
            models.User.role == "student"
        ).order_by(models.User.total_score.desc()).limit(3).all()

        if not top_students:
            print("⚠️ 没有找到学生数据，跳过结算。")
            return

        # 2. 设定前三名的奖励梯度（小红花）
        rewards = [50, 30, 10]
        titles = ["榜首", "榜眼", "探花"]

        for i, student in enumerate(top_students):
            reward_coin = rewards[i]
            rank_title = titles[i]

            # A. 给学生增加小红花
            student.eco_coin += reward_coin

            # B. 写入积分流水记录
            new_record = models.PointRecord(
                user_id=student.id,
                task_type=99,  # 99 代表“排行榜系统发奖”
                points=reward_coin,
                description=f"上周排行榜【{rank_title}】荣誉奖励"
            )
            db.add(new_record)

            # C. 🚀 关键：同步发送首页通知
            new_notice = models.Notification(
                user_id=student.id,
                type="reward",
                content=f"恭喜！你获得了上周排行榜【{rank_title}】，系统奖励了 {reward_coin} 朵小红花，快去商城看看吧！"
            )
            db.add(new_notice)

            print(f"👑 已向 {student.nickname} ({rank_title}) 发放奖励及通知")

        db.commit()
        print("✅ 每周排行榜结算及通知发放圆满完成！")

    except Exception as e:
        print(f"❌ 结算失败: {e}")
        db.rollback()
    finally:
        db.close()


# ==========================================
# 🚀 新版生命周期管理器 Lifespan (替换 on_event)
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 【启动时执行】
    scheduler = BackgroundScheduler()
    # 设定：每周一 (mon) 凌晨 00:00 自动执行
    scheduler.add_job(run_weekly_settlement, 'cron', day_of_week='mon', hour=0, minute=0)
    scheduler.start()
    print("🕰️ APScheduler 定时任务引擎已启动 (Lifespan 模式)。")

    yield  # 这里是应用运行的时间

    # 【关闭时执行】
    scheduler.shutdown()
    print("🛑 APScheduler 定时任务引擎已安全关闭。")


# ==========================================
# 实例化 FastAPI (带上 lifespan)
# ==========================================
app = FastAPI(
    title="智能垃圾分类小程序 API",
    version="1.3",
    lifespan=lifespan
)

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
    ai_model.load_state_dict(torch.load("weights/best_mobilenetv3_1_5.pth", map_location=device, weights_only=True))
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
async def get_admin_menu():
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
                "path": "/schoolClass/index",
                "name": "schoolClass",
                "component": "/schoolClass/index",  # 👈 严格对应 src/views/schoolClass/index.vue
                "meta": {
                    "icon": "OfficeBuilding",
                    "title": "班级架构配置",
                    "isLink": "",
                    "isHide": False,
                    "isFull": False,
                    "isAffix": False,
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
                "path": "/low_confidence",
                "name": "lowConfidenceAudit",
                "component": "/low_confidence/index",  # 前端在 views/low_confidence/index.vue 开发页面
                "meta": {
                    "icon": "Aim",  # 使用一个瞄准或拦截的图标
                    "title": "疑难图片复核",
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
                # 含有子菜单的父级组件通常必须是 "Layout"
                "component": "Layout",
                "meta": {
                    "icon": "User",
                    "title": "小程序用户管理",
                    "isLink": "",
                    "isHide": False,
                    "isFull": False,
                    "isAffix": False,
                    "isKeepAlive": True
                },
                # 嵌套的二级菜单列表
                "children": [
                    {
                        "path": "/users/student/index",
                        "name": "studentManage",
                        "component": "/users/student/index",  # 指向我们刚才新建的小卫士页面
                        "meta": {
                            "icon": "Avatar",  # 小卫士的图标
                            "title": "环保小卫士",
                            "isLink": "",
                            "isHide": False,
                            "isFull": False,
                            "isAffix": False,
                            "isKeepAlive": True
                        }
                    },
                    {
                        "path": "/users/teacher/index",
                        "name": "teacherManage",
                        "component": "/users/teacher/index",  # 指向我们刚才新建的老师页面
                        "meta": {
                            "icon": "Briefcase",  # 老师的图标
                            "title": "指导老师",
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
                "path": "/mall",
                "name": "mallManage",
                "component": "/mall/index",
                "meta": {
                    "icon": "Goods",
                    "title": "小红花商城管理",
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
async def get_admin_buttons():
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

    # 记录修改前的原始状态
    original_status = feedback.status

    feedback.status = req.status
    feedback.admin_reply = req.admin_reply

    # 【逻辑1：图片数据飞轮入库 (仅限图片类型的采纳)】
    if req.status == 1 and feedback.image_url and feedback.type in ["image", "图片"]:
        try:
            raw_suggestion = feedback.suggestion if feedback.suggestion else ""
            correct_category = "未分类"
            for standard_cat in ["可回收物", "有害垃圾", "厨余垃圾", "其他垃圾"]:
                if standard_cat in raw_suggestion:
                    correct_category = standard_cat
                    break

            save_dir = os.path.join("E:/wechat/feedback_image", "train", correct_category)
            os.makedirs(save_dir, exist_ok=True)

            response = requests.get(feedback.image_url, timeout=10)
            if response.status_code == 200:
                file_name = f"feedback_{uuid.uuid4().hex[:8]}.jpg"
                file_path = os.path.join(save_dir, file_name)
                with open(file_path, "wb") as f:
                    f.write(response.content)
                print(f"飞轮运转：已成功将纠错照片采纳入训练集 -> {file_path}")
        except Exception as e:
            print(f"采纳图片处理时发生异常: {e}")

    # ==========================================
    # 【逻辑2：独立的环保币发放/扣回逻辑】
    # ==========================================

    # 场景A：首次被采纳 (无论图片还是文本纠错，都给钱)
    if req.status == 1 and original_status != 1:
        user = db.query(models.User).filter(models.User.id == feedback.user_id).first()
        if user:
            user.eco_coin += 2  # 增加 2 朵小红花

            new_record = models.PointRecord(
                user_id=user.id,
                change_amount=2,
                task_type=6,  # 约定 6 代表纠错奖励
                description=f"纠错被采纳奖励：{feedback.item_name}"
            )
            db.add(new_record)

    # 场景B：管理员误操作点成采纳，后又修改为驳回 (需要把错发的钱扣回来)
    elif req.status == 2 and original_status == 1:
        user = db.query(models.User).filter(models.User.id == feedback.user_id).first()
        if user:
            user.eco_coin -= 2  # 追回 2 朵小红花

            new_record = models.PointRecord(
                user_id=user.id,
                change_amount=-2,
                task_type=6,
                description=f"纠错重新驳回，扣除奖励：{feedback.item_name}"
            )
            db.add(new_record)

    db.commit()
    return {"code": 200, "message": "审核完成，数据状态已同步", "data": None}

# ==============================================================================
# 后台管理系统 - 小程序用户管理模块
# ==============================================================================

@app.get("/api/admin/users")
async def get_admin_users(
    pageNum: int = Query(1, description="当前页码"),
    pageSize: int = Query(10, description="每页数量"),
    nickname: str = Query(None, description="搜索：用户昵称"),
    role: str = Query(None, description="搜索：用户角色"),
    class_id: int = Query(None, description="搜索：所属班级ID"),  # 核心修复 1：接收前端传来的 class_id
    db: Session = Depends(get_db)
):
    """
    分页获取小程序注册用户列表
    """
    query = db.query(models.User)

    # 1. 支持按昵称模糊搜索
    if nickname:
        query = query.filter(models.User.nickname.like(f"%{nickname}%"))

    # 2. 支持按角色精准筛选
    if role:
        query = query.filter(models.User.role == role)

    # 核心修复 2：支持按班级精准筛选
    if class_id:
        query = query.filter(models.User.class_id == class_id)

    total = query.count()
    skip = (pageNum - 1) * pageSize
    users = query.order_by(models.User.id.desc()).offset(skip).limit(pageSize).all()

    list_data = []
    for u in users:
        # 获取班级名字
        c_name = f"{u.school_class.grade_name} {u.school_class.class_name}" if u.school_class else "未分配"

        list_data.append({
            "id": u.id,
            "openid": u.openid,
            "nickname": u.nickname or "微信用户",
            "avatar_url": u.avatar_url or "",
            "role": u.role,
            "class_id": u.class_id,  # 顺手修复：返回 class_id，以便前端点击“编辑”时下拉框能自动选中当前班级
            "class_info": c_name,
            "title": u.title,
            "score": u.total_score,
            "eco_coin": u.eco_coin,
            "created_at": u.created_at.strftime("%Y-%m-%d") if u.created_at else ""
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
# 后台管理系统 - 班级管理模块
# ==============================================================================
from pydantic import BaseModel
class AdminClassSchema(BaseModel):
    grade_name: str
    class_name: str

# 1. 管理员获取所有班级列表
@app.get("/api/admin/classes")
async def get_admin_classes(db: Session = Depends(get_db)):
    classes = db.query(models.SchoolClass).all()
    # 拼装给前端
    list_data = [{"id": c.id, "grade_name": c.grade_name, "class_name": c.class_name} for c in classes]
    return {"code": 200, "message": "成功", "data": list_data}

# 2. 管理员新增班级
@app.post("/api/admin/classes")
async def add_admin_class(req: AdminClassSchema, db: Session = Depends(get_db)):
    new_class = models.SchoolClass(grade_name=req.grade_name, class_name=req.class_name)
    db.add(new_class)
    db.commit()
    return {"code": 200, "message": "班级添加成功"}

# --- 数据校验 Schema ---
class AuditLowConfidenceSchema(BaseModel):
    id: int
    status: int  # 1-打标入库, 2-废弃(比如用户拍了张纯黑的废图)
    correct_category_name: Optional[str] = None  # 如果入库，管理员选择的真实四大类名称


# --- 接口：获取低置信度拦截列表 ---
@app.get("/api/admin/low_confidence")
async def get_low_confidence_list(
        pageNum: int = Query(1),
        pageSize: int = Query(10),
        status: int = Query(None),
        db: Session = Depends(get_db)
):
    query = db.query(models.LowConfidenceRecord)
    if status is not None:
        query = query.filter(models.LowConfidenceRecord.status == status)

    total = query.count()
    skip = (pageNum - 1) * pageSize
    records = query.order_by(models.LowConfidenceRecord.created_at.desc()).offset(skip).limit(pageSize).all()

    list_data = [{"id": r.id, "image_url": r.image_url, "confidence": r.confidence, "status": r.status} for r in
                 records]

    return {"code": 200, "data": {"list": list_data, "total": total, "pageNum": pageNum, "pageSize": pageSize}}


# --- 接口：打标并让数据飞轮运转 ---
@app.post("/api/admin/low_confidence/audit")
async def audit_low_confidence(req: AuditLowConfidenceSchema, db: Session = Depends(get_db)):
    record = db.query(models.LowConfidenceRecord).filter(models.LowConfidenceRecord.id == req.id).first()
    if not record:
        return {"code": 404, "message": "记录不存在"}

    record.status = req.status

    # 【核心逻辑：真实打标入库训练集】
    if req.status == 1 and req.correct_category_name:
        try:
            # 存入你之前写好的训练集目录
            save_dir = os.path.join("E:/wechat/feedback_image", "train", req.correct_category_name)
            os.makedirs(save_dir, exist_ok=True)

            response = requests.get(record.image_url, timeout=10)
            if response.status_code == 200:
                file_name = f"hard_example_{uuid.uuid4().hex[:8]}.jpg"
                file_path = os.path.join(save_dir, file_name)
                with open(file_path, "wb") as f:
                    f.write(response.content)
                print(f"难例飞轮运转：已将低置信度照片打标入库 -> {file_path}")
        except Exception as e:
            print(f"下载难例图片失败: {e}")

    db.commit()
    return {"code": 200, "message": "处理完成"}

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

# ==============================================================================
# 后台管理系统 - 积分商城管理
# ==============================================================================
class AdminMallItemSchema(BaseModel):
    name: str
    desc: str
    points_price: int
    image_url: str
    stock: int
    is_active: bool


# --- 1. 获取商品列表 (带分页和搜索) ---
@app.get("/api/admin/mall/items")
async def get_admin_mall_items(
        pageNum: int = Query(1),
        pageSize: int = Query(10),
        name: str = Query(None),
        db: Session = Depends(get_db)
):
    query = db.query(models.MallItem)
    if name:
        query = query.filter(models.MallItem.name.like(f"%{name}%"))

    total = query.count()
    skip = (pageNum - 1) * pageSize
    items = query.order_by(models.MallItem.id.desc()).offset(skip).limit(pageSize).all()

    list_data = []
    for i in items:
        list_data.append({
            "id": i.id,
            "name": i.name,
            "desc": i.desc,
            "points_price": i.points_price,
            "image_url": i.image_url,
            "stock": i.stock,
            "is_active": i.is_active,
            "created_at": i.created_at.strftime("%Y-%m-%d %H:%M:%S") if i.created_at else ""
        })

    return {"code": 200, "message": "成功",
            "data": {"list": list_data, "total": total, "pageNum": pageNum, "pageSize": pageSize}}


# --- 2. 新增商品 ---
@app.post("/api/admin/mall/items")
async def add_admin_mall_item(item_data: AdminMallItemSchema, db: Session = Depends(get_db)):
    new_item = models.MallItem(**item_data.dict())
    db.add(new_item)
    db.commit()
    return {"code": 200, "message": "新增成功", "data": None}


# --- 3. 修改商品 ---
@app.put("/api/admin/mall/items/{item_id}")
async def edit_admin_mall_item(item_id: int, item_data: AdminMallItemSchema, db: Session = Depends(get_db)):
    item = db.query(models.MallItem).filter(models.MallItem.id == item_id).first()
    if not item: return {"code": 404, "message": "商品不存在"}

    for key, value in item_data.dict().items():
        setattr(item, key, value)

    db.commit()
    return {"code": 200, "message": "修改成功", "data": None}


# --- 4. 批量删除商品 ---
class AdminDeleteMallSchema(BaseModel):
    id: list[int]


@app.post("/api/admin/mall/items/delete")
async def delete_admin_mall_items(req: AdminDeleteMallSchema, db: Session = Depends(get_db)):
    db.query(models.MallItem).filter(models.MallItem.id.in_(req.id)).delete(synchronize_session=False)
    db.commit()
    return {"code": 200, "message": "删除成功", "data": None}


# 1. 定义 Pydantic Schema 用于接收参数
class RefundSchema(BaseModel):
    user_id: int
    redemption_id: int


# ==========================================
# 处理退货反悔逻辑 (严格事务防刷)
# ==========================================
@app.post("/api/mall/refund")
async def refund_mall_item(req: RefundSchema, db: Session = Depends(get_db)):
    # 1. 查出该订单记录
    record = db.query(models.RedemptionRecord).filter(
        models.RedemptionRecord.id == req.redemption_id,
        models.RedemptionRecord.user_id == req.user_id
    ).first()

    if not record:
        return {"code": 404, "message": "该兑换记录不存在"}

    # 2. 核心逻辑校验：只有“待核销”状态才能反悔
    if record.status != 0:
        if record.status == 1:
            return {"code": 400, "message": "商品已核销发货，无法退货哦。可以找老师线下沟通"}
        else:
            return {"code": 400, "message": "该订单已是退货状态，无法重复操作"}

    # ============= 开始事务操作 =============
    try:
        # 3. 查人、查商品
        user = db.query(models.User).filter(models.User.id == req.user_id).first()
        item = db.query(models.MallItem).filter(models.MallItem.id == record.item_id).first()

        if not user or not item:
            raise Exception("用户或商品已不在，事务回滚")

        # 4. 原路退回用户【环保币】
        user.eco_coin += record.points_cost

        # 5. 写入积分流水账单 (task_type=5 为商城退款)
        point_record = models.PointRecord(
            user_id=user.id,
            change_amount=record.points_cost,  # 退款是正数
            task_type=5,
            description=f"兑换反悔退款：{item.name}"
        )
        db.add(point_record)

        # 6. 将订单状态标记为 2-已退货
        record.status = 2
        # 可选：记录退款时间
        # record.updated_at = datetime.now()

        # 7. 退回真实库存
        if item.stock != -1:  # 如果是限量实体商品
            item.stock += 1

        db.commit()

        return {
            "code": 200,
            "message": "反悔成功！积分已原路退回",
            "data": {
                "new_score": user.total_score,
                "new_title": user.title
            }
        }
    except Exception as e:
        db.rollback()
        print("退货发生异常：", e)
        return {"code": 500, "message": "服务器开小差了，反悔失败"}

@app.get("/")
def read_root():
    return {"message": "垃圾分类后端服务已成功启动！"}


# 定义注册请求接收的数据格式
class WxRegisterRequest(BaseModel):
    openid: str
    role: str = "student"
    invite_code: Optional[str] = None  # 选填，但如果是 teacher 则必填
    nickname: str = "微信用户"
    avatar_url: str = ""
    class_id: int = 1

class UpdateClassSchema(BaseModel):
    user_id: int
    class_id: int

# ==========================================
# 接口：微信静默登录
# ==========================================
@app.post("/api/user/login")
def wechat_login(request_data: schemas.WxLoginRequest, db: Session = Depends(get_db)):
    """
    静默登录：只换取 OpenID 并查库。
    如果有：返回 200 和角色信息直接进首页。
    如果没有：返回 404 和 OpenID，让前端拦截去注册页。
    """
    url = f"https://api.weixin.qq.com/sns/jscode2session?appid={WX_APPID}&secret={WX_SECRET}&js_code={request_data.code}&grant_type=authorization_code"

    response = requests.get(url)
    res_data = response.json()

    if "errcode" in res_data and res_data["errcode"] != 0:
        return {"code": 400, "message": f"微信授权失败: {res_data.get('errmsg')}", "data": None}

    openid = res_data.get("openid")
    if not openid:
        return {"code": 400, "message": "未获取到有效 OpenID", "data": None}

    # 查库看是否已经是老用户
    user = db.query(models.User).filter(models.User.openid == openid).first()

    if user:
        # 老用户，静默登录成功
        grade_name = user.school_class.grade_name if user.school_class else "未分配"
        class_name = user.school_class.class_name if user.school_class else ""
        return {
            "code": 200,
            "message": "静默登录成功",
            "data": {
                "id": user.id,
                "openid": user.openid,
                "role": user.role,
                "nickname": user.nickname,
                "avatar_url": user.avatar_url,
                "full_class_name": f"{grade_name} {class_name}".strip()  # 拼好给前端展示
            }
        }
    else:
        # 新用户，不再自动落库！返回 404 让前端去注册页
        return {
            "code": 404,
            "message": "新用户未注册，请前往身份选择页",
            "data": {
                "openid": openid  # 把 openid 吐给前端，前端在下一步注册时要传回来
            }
        }


# ==========================================
# 接口：新用户角色注册
# ==========================================
@app.post("/api/user/register")
def wechat_register(req: WxRegisterRequest, db: Session = Depends(get_db)):
    """
    前端选择身份后，调用此接口真正完成注册落库。
    包含教师邀请码的安全拦截逻辑。
    """
    # 1. 防止重复注册
    exist_user = db.query(models.User).filter(models.User.openid == req.openid).first()
    if exist_user:
        return {"code": 400, "message": "该微信账号已注册，请直接登录"}

    # 2. 如果选择老师角色，必须校验邀请码
    invite_record = None
    if req.role == "teacher":
        if not req.invite_code:
            return {"code": 400, "message": "注册老师账号需要提供专属邀请码哦"}

        # 查邀请码表
        invite_record = db.query(models.TeacherInviteCode).filter(
            models.TeacherInviteCode.code == req.invite_code).first()

        if not invite_record:
            return {"code": 400, "message": "邀请码无效，请联系管理员获取"}
        if invite_record.is_used:
            return {"code": 400, "message": "该邀请码已经被其他老师使用过了"}

    # 3. 校验通过，创建新用户
    new_user = models.User(
        openid=req.openid,
        role=req.role,
        nickname=req.nickname,
        avatar_url=req.avatar_url,
        class_id=req.class_id  # 新增入库
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    # 4. 如果是老师，立即将该邀请码核销并绑定使用者
    if req.role == "teacher" and invite_record:
        invite_record.is_used = True
        invite_record.used_by = new_user.id
        db.commit()

    # 5. 手动拼接班级名称返回给前端
    # 这里访问 new_user.school_class 会自动触发 SQLAlchemy 去班级表里查名字
    grade_name = new_user.school_class.grade_name if new_user.school_class else "未分配"
    class_name = new_user.school_class.class_name if new_user.school_class else ""
    full_name = f"{grade_name} {class_name}".strip()

    # 组装完整的数据包返回
    return {
        "code": 200,
        "message": "注册成功！欢迎加入",
        "data": {
            "id": new_user.id,
            "openid": new_user.openid,  # 顺手补齐，保持和 login 接口一致
            "role": new_user.role,
            "nickname": new_user.nickname,
            "avatar_url": new_user.avatar_url,  # 顺手补齐
            "full_class_name": full_name
        }
    }


# 修改班级的专属接口
@app.post("/api/user/update_class")
def update_user_class(req: UpdateClassSchema, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.id == req.user_id).first()
    if not user: return {"code": 404, "message": "用户不存在"}

    user.class_id = req.class_id
    db.commit()

    # 重新查一遍返回最新的班级名称
    db.refresh(user)
    return {
        "code": 200,
        "message": "班级修改成功",
        "data": {"full_class_name": f"{user.school_class.grade_name} {user.school_class.class_name}"}
    }
# ==========================================
# 处理每日任务积分奖励与防刷校验
# ==========================================
def check_and_award_daily_task(user_id: int, task_type: int, reward_amount: int, description: str, db: Session) -> int:
    """
    通用防刷奖励机制
    :return: 实际获得的积分（若今天已完成则返回 0）
    """
    user = db.query(models.User).filter(models.User.id == user_id).first()

    # 如果是老师，直接阻断积分发放源头
    if not user or user.role == "teacher":
        return 0
    today = date.today()

    # 1. 查询流水表，检查今天是否已完成该类型任务
    daily_record = db.query(models.PointRecord).filter(
        models.PointRecord.user_id == user_id,
        models.PointRecord.task_type == task_type,
        func.date(models.PointRecord.created_at) == today
    ).first()

    # 2. 如果今天没做过，发放奖励
    if not daily_record:
        # 写入积分流水
        new_record = models.PointRecord(
            user_id=user_id,
            change_amount=reward_amount,
            task_type=task_type,
            description=description
        )
        db.add(new_record)

        # 日常任务只增加环保币 (eco_coin)
        user = db.query(models.User).filter(models.User.id == user_id).first()
        if user:
            user.eco_coin += reward_amount  # 只加环保币

        db.commit()
        return reward_amount

    # 今天已经做过了，返回 0
    return 0

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

    # 低置信度难例自动拦截入库逻辑
    CONFIDENCE_THRESHOLD = 70.0  # 设置阈值
    if conf_val < CONFIDENCE_THRESHOLD:  # 使用 conf_val (85.11) 进行比较
        new_hard_example = models.LowConfidenceRecord(
            image_url=cos_image_url,
            ai_predicted_category=category_info.id,
            confidence=conf_val  # 将普通的 float 存入数据库
        )
        db.add(new_hard_example)

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

    # 触发每日首次拍照奖励 (task_type=1)
    reward_points = check_and_award_daily_task(
        user_id=user_id,
        task_type=1,
        reward_amount=1,
        description="每日首次拍照打卡奖励",
        db=db
    )

    # 组装给前端的返回结果（加入奖励积分）
    mock_result = {
        "user_id": user_id,
        "image_path": cos_image_url,
        "confidence": conf_val,
        "category_id": category_info.id,
        "category_name": category_info.category_name,
        "category_class": category_info.category_class,

        "eco_value": category_info.eco_value,
        "put_guidance": category_info.put_guidance,
        "harm_description": category_info.harm_description,
        "process_method": category_info.process_method,
        "sub_guidance": category_info.sub_guidance,

        "recommend_items": recommend_list,
        "reward_points": reward_points  # 把获得的积分传给前端
    }

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
    # 低置信度难例自动拦截入库逻辑
    CONFIDENCE_THRESHOLD = 70.0  # 设置阈值
    if confidence < CONFIDENCE_THRESHOLD:
        new_hard_example = models.LowConfidenceRecord(
            image_url=cos_image_url,
            ai_predicted_category=category_info.id,
            confidence=confidence
        )
        db.add(new_hard_example)

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

    reward_points = check_and_award_daily_task(
        user_id=user_id,
        task_type=1,
        reward_amount=1,
        description="每日首次拍照打卡奖励",
        db=db
    )

    # 4. 组装结果返回给前端展示
    mock_result = {
        "user_id": user_id,
        "image_path": cos_image_url,
        "confidence": confidence,
        "category_id": category_info.id,
        "category_name": category_info.category_name,
        "category_class": category_info.category_class,

        "eco_value": category_info.eco_value,
        "put_guidance": category_info.put_guidance,
        "harm_description": category_info.harm_description,
        "process_method": category_info.process_method,
        "sub_guidance": category_info.sub_guidance,

        "recommend_items": recommend_list,
        "reward_points": reward_points
    }

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


from pydantic import BaseModel


class ReadTaskSchema(BaseModel):
    user_id: int


# ==========================================
# 完成每日阅读科普任务打卡
# ==========================================
@app.post("/api/task/read_tip")
async def finish_read_task(req: ReadTaskSchema, db: Session = Depends(get_db)):
    # 直接复用我们上次写的“瑞士军刀”函数！
    # task_type=2 代表阅读打卡，奖励 15 积分
    reward_points = check_and_award_daily_task(
        user_id=req.user_id,
        task_type=2,
        reward_amount=1,
        description="每日科普阅读打卡奖励",
        db=db
    )

    # 顺便查一下用户最新总积分返回去
    user = db.query(models.User).filter(models.User.id == req.user_id).first()

    return {
        "code": 200,
        "message": "阅读记录成功",
        "data": {
            "reward_points": reward_points,
            "total_score": user.total_score if user else 0,
            "title": user.title if user else "环保新手"
        }
    }

# --- 记录阅读行为的接口 ---
# 1. 定义一个用于接收前端 JSON Body 的数据模型
class ReadingRecordCreate(BaseModel):
    user_id: int
    tip_id: int

# 2. 修改接口，使用刚刚定义的模型来接收数据
@app.post("/api/user/reading/record")
async def add_reading_record(record: ReadingRecordCreate, db: Session = Depends(get_db)):
    # 3. 从模型中提取数据并保存
    new_record = models.ReadingRecord(user_id=record.user_id, tip_id=record.tip_id)
    db.add(new_record)
    db.commit()
    return {"code": 200, "message": "阅读记录已保存"}

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
            "image_url": item.image_url,
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

    # ==========================================
    # 后端防抖与防重复交卷拦截
    # ==========================================
    # 检查该用户在过去 5 秒内，是否提交过完全相同的成绩
    five_seconds_ago = datetime.now() - timedelta(seconds=5)
    duplicate_record = db.query(models.ChallengeHistory).filter(
        models.ChallengeHistory.user_id == quiz_data.user_id,
        models.ChallengeHistory.correct_count == quiz_data.correct_count,
        models.ChallengeHistory.total_count == quiz_data.total_count,
        models.ChallengeHistory.created_at >= five_seconds_ago
    ).first()

    if duplicate_record:
        # 如果 5 秒内查到了同样的记录，说明是网络延迟导致的连击，直接驳回！
        return {"code": 400, "message": "请勿频繁重复交卷哦"}
    # ==========================================
    # 根据双模式规则，后端权威计算得分
    # ==========================================
    calculated_score = 0
    accuracy = quiz_data.correct_count / quiz_data.total_count if quiz_data.total_count > 0 else 0
    current_performance = "再接再厉"

    if quiz_data.mode == "timed":
        # 基础奖励：只要参与并交卷，固定获得 1 颗环保星
        base_score = 1
        bonus = 0

        # 计算核心判定指标
        is_on_time = quiz_data.time_used <= quiz_data.total_time
        accuracy = quiz_data.correct_count / quiz_data.total_count if quiz_data.total_count > 0 else 0
        progress = quiz_data.total_count / 25  # 假设限时模式目标总题数为 25

        # 🚀 场景化评价引擎
        if is_on_time:
            if accuracy >= 0.8:
                # 场景 1：按时完成且正确率高
                bonus = 3
                current_performance = "手速王者"
            else:
                # 场景 2：按时完成但正确率不高
                bonus = 2
                current_performance = "极速达人"
        else:
            if progress >= 0.6:
                if accuracy >= 0.8:
                    # 场景 3：未按时完成，但答题大半且正确率高
                    bonus = 1
                    current_performance = "沉稳小将"
                else:
                    # 场景 4：未按时完成，答题大半，但正确率不高
                    bonus = 0
                    current_performance = "游刃有余"
            else:
                # 场景 5：未按时完成且答题未过半
                bonus = 0
                current_performance = "眼疾手快"

        # 最终得分 = 基础1星 + 加成星
        calculated_score = base_score + bonus

    else:
        # 【经典模式】基础星 1
        base_score = 1
        bonus = 0

        if accuracy == 1.0:
            bonus = 1  # 全对额外拿1星
            current_performance = "完美通关"
        elif accuracy >= 0.8:
            current_performance = "火眼金睛"
        elif accuracy >= 0.5:
            current_performance = "渐入佳境"
        else:
            current_performance = "再接再厉"

        calculated_score = base_score + bonus

    # ==========================================
    # 更新用户总星数和童趣版环保称号 (降低门槛)
    # ==========================================
    user.total_score += calculated_score

    new_title = user.title
    if user.total_score >= 50:
        new_title = "环保小宗师"  # 原为 500
    elif user.total_score >= 20:
        new_title = "环保小达人"  # 原为 200
    elif user.total_score >= 5:
        new_title = "环保小卫士"  # 原为 50
    else:
        new_title = "环保小萌新"
    user.title = new_title

    # 存入历史记录
    new_history = models.ChallengeHistory(
        user_id=user.id,
        score=calculated_score,
        correct_count=quiz_data.correct_count,
        mode=quiz_data.mode,
        total_count=quiz_data.total_count
    )
    db.add(new_history)

    # 存入错题本
    for wrong_item in quiz_data.wrong_answers:
        new_wrong = models.WrongBook(
            user_id=user.id,
            item_name=wrong_item.item_name,
            user_answer=wrong_item.user_answer,
            correct_answer=wrong_item.correct_answer,
            status=0 # 默认未掌握
        )
        db.add(new_wrong)

    # ==========================================
    # 更新用户四大分类的雷达图学情库
    # ==========================================
    if hasattr(quiz_data, 'category_stats') and quiz_data.category_stats:
        for stat in quiz_data.category_stats:
            # 只有当这次挑战中确实遇到了该类别的题目时，才更新数据库
            if stat.total > 0:
                # 查询该用户该分类的统计记录是否已存在
                stat_record = db.query(models.UserCategoryStat).filter(
                    models.UserCategoryStat.user_id == user.id,
                    models.UserCategoryStat.category_type == stat.category_type
                ).first()

                if stat_record:
                    # 老记录：直接累加（这非常快）
                    stat_record.total_answered += stat.total
                    stat_record.correct_answered += stat.correct
                else:
                    # 新记录：如果小朋友是第一次做这类题，创建新行
                    new_stat = models.UserCategoryStat(
                        user_id=user.id,
                        category_type=stat.category_type,
                        total_answered=stat.total,
                        correct_answered=stat.correct
                    )
                    db.add(new_stat)

    db.commit()

    # 每日首次挑战奖励 (奖励 2 小红花)
    reward_eco_coin = check_and_award_daily_task(
        user_id=user.id, task_type=3, reward_amount=2,
        description="每日首次挑战打卡奖励", db=db
    )

    return {
        "code": 200,
        "message": "交卷成功！",
        "data": {
            "earned_score": calculated_score,  # 将真实得分返回给前端展示
            "total_score": user.total_score,
            "current_title": user.title,
            "performance": current_performance,
            "reward_eco_coin": reward_eco_coin
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

    # 加上 is_deleted == False，保证用户主页的统计数字和列表里看到的一致
    recognize_count = db.query(models.RecognizeHistory).filter(
        models.RecognizeHistory.user_id == user_id,
        models.RecognizeHistory.is_deleted == False
    ).count()

    challenge_count = db.query(models.ChallengeHistory).filter(
        models.ChallengeHistory.user_id == user_id,
        models.ChallengeHistory.is_deleted == False
    ).count()

    return {
        "code": 200,
        "data": {
            "total_score": user.total_score,
            "eco_coin": user.eco_coin,
            "title": user.title,
            "recognize_count": recognize_count,
            "challenge_count": challenge_count
        }
    }


# 2. 获取识别历史列表
@app.get("/api/user/recognize_history")
async def get_recognize_history(user_id: int, db: Session = Depends(get_db)):
    histories = db.query(models.RecognizeHistory).filter(
        models.RecognizeHistory.user_id == user_id,
        models.RecognizeHistory.is_deleted == False
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

# 3. 获取挑战历史列表 (通过时间戳智能匹配错题本)
@app.get("/api/user/challenge_history")
async def get_challenge_history(user_id: int, db: Session = Depends(get_db)):
    histories = db.query(models.ChallengeHistory).filter(
        models.ChallengeHistory.user_id == user_id,
        models.ChallengeHistory.is_deleted == False
    ).order_by(desc(models.ChallengeHistory.created_at)).all()

    result = []
    for h in histories:
        # 兼容老数据（如果老数据没这两个字段，默认为经典模式）
        mode = getattr(h, 'mode', 'classic')
        total_count = getattr(h, 'total_count', 10)

        # 针对不同模式，采用不同的称号评级逻辑
        if mode == 'timed':
            if h.correct_count >= 20:
                perf_text, t_class = "手速王者", "level-4"
            elif h.correct_count >= 15:
                perf_text, t_class = "极速达人", "level-3"
            elif h.correct_count >= 10:
                perf_text, t_class = "游刃有余", "level-2"
            else:
                perf_text, t_class = "眼疾手快", "level-1"
        else:
            if h.correct_count >= 10:
                perf_text, t_class = "完美通关", "level-4"
            elif h.correct_count >= 8:
                perf_text, t_class = "火眼金睛", "level-3"
            elif h.correct_count >= 5:
                perf_text, t_class = "渐入佳境", "level-2"
            else:
                perf_text, t_class = "再接再厉", "level-1"

        # 通过 created_at 匹配当时产生的错题记录
        wrong_list_formatted = []
        if h.created_at:
            # 设定前后 2 秒的时间窗口，防止跨表写入的毫秒级延迟导致查不到
            time_lower = h.created_at - timedelta(seconds=2)
            time_upper = h.created_at + timedelta(seconds=2)

            matched_wrongs = db.query(models.WrongBook).filter(
                models.WrongBook.user_id == user_id,
                models.WrongBook.created_at >= time_lower,
                models.WrongBook.created_at <= time_upper,
                models.WrongBook.is_deleted == False  # 严谨起见，过滤掉已被逻辑删除的错题
            ).all()

            for w in matched_wrongs:
                wrong_list_formatted.append({
                    "name": w.item_name,
                    "userSelect": w.user_answer,
                    "correctAnswer": w.correct_answer
                })

        result.append({
            "id": str(h.id),
            "score": h.score,
            "correctCount": h.correct_count,
            "totalCount": total_count,
            "mode": mode,
            "title": perf_text,
            "titleClass": t_class,
            "date": h.created_at.strftime("%Y-%m-%d %H:%M") if h.created_at else "",
            "wrongList": wrong_list_formatted
        })

    return {"code": 200, "data": result}


# 4. 获取我的错题本
@app.get("/api/user/wrong_book")
async def get_wrong_book(user_id: int, db: Session = Depends(get_db)):
    # 1. 查出该用户所有未被删除的错题，按时间倒序
    histories = db.query(models.WrongBook).filter(
        models.WrongBook.user_id == user_id,
        models.WrongBook.is_deleted == False
    ).order_by(desc(models.WrongBook.created_at)).all()

    # 2. 核心逻辑：利用字典进行“读时聚合”去重并统计次数
    unique_wrongs = {}
    for w in histories:
        if w.item_name not in unique_wrongs:
            # 如果是第一次遍历到这个垃圾（因为是时间倒序，这一定是最近一次做错的记录）
            unique_wrongs[w.item_name] = {
                "id": str(w.id),
                "name": w.item_name,
                "userSelect": w.user_answer,
                "correctAnswer": w.correct_answer,
                "errorCount": 1,

                # 必须把这俩字段传给前端！否则前端无法判断状态和时间！
                "status": w.status,
                "created_at": w.created_at.isoformat() if w.created_at else None
            }
        else:
            # 如果字典里已经有了，说明历史上也错过，只增加错误次数
            unique_wrongs[w.item_name]["errorCount"] += 1

            # 💡 细节优化：即使多次做错，如果最新的这一次状态是“已消灭(1)”，也应该保留
            # （因为历史是倒序遍历的，第一次遇到的就是最新的）

    # 3. 将字典的值转化为列表返回给前端
    result = list(unique_wrongs.values())

    return {"code": 200, "data": result}


# 7. 获取反馈历史列表
@app.get("/api/user/feedback_history")
async def get_feedback_history(user_id: int, db: Session = Depends(get_db)):
    histories = db.query(models.Feedback).filter(
        models.Feedback.user_id == user_id,
        models.Feedback.is_deleted == False
    ).order_by(desc(models.Feedback.created_at)).all()

    result = []
    for f in histories:
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

# 8. 获取我的兑换记录 (联表查询商城商品)
@app.get("/api/user/redemption_history")
async def get_redemption_history(user_id: int, db: Session = Depends(get_db)):
    # 联表查询 RedemptionRecord 和 MallItem
    records = db.query(models.RedemptionRecord, models.MallItem).join(
        models.MallItem, models.RedemptionRecord.item_id == models.MallItem.id
    ).filter(
        models.RedemptionRecord.user_id == user_id
    ).order_by(desc(models.RedemptionRecord.created_at)).all()

    result = []
    for record, item in records:
        result.append({
            "id": str(record.id),
            "itemName": item.name,
            "imageUrl": item.image_url if item.image_url else "/images/default_item.png",
            "pointsCost": record.points_cost,
            "status": record.status,  # 0:待核销, 1:已完成
            "date": record.created_at.strftime("%Y-%m-%d %H:%M") if record.created_at else ""
        })
    return {"code": 200, "data": result}

# ==============================================================================
# 所有清空 (clear) 接口必须定义在单条删除 ({item_id}) 接口上方
# ==============================================================================

# --- 错题本 删除接口 ---
@app.delete("/api/user/wrong_book/clear")
async def clear_wrong_book(user_id: int, db: Session = Depends(get_db)):
    db.query(models.WrongBook).filter(
        models.WrongBook.user_id == user_id,
        models.WrongBook.is_deleted == False
    ).update({"is_deleted": True})
    db.commit()
    return {"code": 200, "message": "清空成功"}


@app.delete("/api/user/wrong_book/{item_id}")
async def delete_wrong_item(item_id: int, db: Session = Depends(get_db)):
    item = db.query(models.WrongBook).filter(models.WrongBook.id == item_id).first()
    if item:
        item.is_deleted = True
        db.commit()
    return {"code": 200, "message": "删除成功"}


# --- 识别历史 删除接口 ---
@app.delete("/api/user/recognize_history/clear")
async def clear_recognize_history(user_id: int, db: Session = Depends(get_db)):
    db.query(models.RecognizeHistory).filter(
        models.RecognizeHistory.user_id == user_id,
        models.RecognizeHistory.is_deleted == False
    ).update({"is_deleted": True})
    db.commit()
    return {"code": 200, "message": "清空成功"}


@app.delete("/api/user/recognize_history/{item_id}")
async def delete_recognize_history(item_id: int, db: Session = Depends(get_db)):
    item = db.query(models.RecognizeHistory).filter(models.RecognizeHistory.id == item_id).first()
    if item:
        item.is_deleted = True
        db.commit()
    return {"code": 200, "message": "删除成功"}


# --- 挑战历史 删除接口 (仅隐藏记录，保留用户真实积分与段位) ---
@app.delete("/api/user/challenge_history/clear")
async def clear_challenge_history(user_id: int, db: Session = Depends(get_db)):
    # 1. 逻辑删除所有挑战记录
    db.query(models.ChallengeHistory).filter(
        models.ChallengeHistory.user_id == user_id,
        models.ChallengeHistory.is_deleted == False
    ).update({"is_deleted": True})
    db.commit()

    # 2. 获取用户当前积分(不扣分)，返回给前端，防止前端报错
    user = db.query(models.User).filter(models.User.id == user_id).first()
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
    # 1. 查询该条记录
    history = db.query(models.ChallengeHistory).filter(models.ChallengeHistory.id == item_id).first()
    if not history:
        return {"code": 404, "message": "记录不存在"}

    # 2. 逻辑删除，不再扣除历史分数
    history.is_deleted = True
    db.commit()

    # 3. 将用户当前的真实积分和称号返回给前端
    user = db.query(models.User).filter(models.User.id == history.user_id).first()
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
    # 只清理 status != 0（已采纳或已驳回）的记录，保护待处理记录！
    db.query(models.Feedback).filter(
        models.Feedback.user_id == user_id,
        models.Feedback.is_deleted == False,
        models.Feedback.status != 0  # 新增的保护条件
    ).update({"is_deleted": True})

    db.commit()
    return {"code": 200, "message": "已清空办结记录"}


@app.delete("/api/user/feedback_history/{item_id}")
async def delete_feedback_history(item_id: int, db: Session = Depends(get_db)):
    item = db.query(models.Feedback).filter(models.Feedback.id == item_id).first()
    if item:
        item.is_deleted = True
        db.commit()
    return {"code": 200, "message": "删除成功"}

# 排行榜接口
@app.get("/api/leaderboard")
def get_leaderboard(user_id: int = Query(...), db: Session = Depends(get_db)): # 👈 新增入参
    user = db.query(models.User).filter(models.User.id == user_id).first()
    target_class_id = user.class_id if user else 1

    # 只排同班同学的
    top_users = db.query(models.User).filter(
        models.User.role == "student",
        models.User.class_id == target_class_id
    ).order_by(models.User.total_score.desc()).limit(10).all()

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


class RedeemSchema(BaseModel):
    user_id: int
    item_id: int


# ==========================================
# 小程序接口：获取商城在售商品列表
# ==========================================
@app.get("/api/mall/items")
async def get_mall_items(user_id: int = Query(...), db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    target_class_id = user.class_id if user else 1

    # 🚀 核心修改：增加 is_deleted == False 过滤条件，屏蔽被删除的商品
    items = db.query(models.MallItem).filter(
        models.MallItem.is_active == True,
        models.MallItem.is_deleted == False,  # 👈 关键点：排除已逻辑删除的商品
        ((models.MallItem.class_id == target_class_id) | (models.MallItem.class_id == 1))
    ).order_by(models.MallItem.points_price.asc()).all()

    result = []
    for item in items:
        result.append({
            "id": item.id,
            "name": item.name,
            "desc": item.desc,
            "points": item.points_price,
            "image": item.image_url if item.image_url else "/images/default_item.png",
            "stock": item.stock
        })
    return {"code": 200, "message": "获取成功", "data": result}


# ==========================================
# 核心兑换逻辑 (防超卖、防扣负)
# ==========================================
@app.post("/api/mall/redeem")
async def redeem_mall_item(req: RedeemSchema, db: Session = Depends(get_db)):
    # 1. 查人、查商品
    user = db.query(models.User).filter(models.User.id == req.user_id).first()
    item = db.query(models.MallItem).filter(models.MallItem.id == req.item_id).first()

    if not user or not item:
        return {"code": 404, "message": "用户或商品不存在"}

    if not item.is_active:
        return {"code": 400, "message": "该商品已下架"}

    # 2. 检查库存 (如果是 -1 则代表无限)
    if item.stock == 0:
        return {"code": 400, "message": "手慢啦，商品已兑换完"}

    # 3. 检查【环保币】是否足够
    if user.eco_coin < item.points_price:
        return {"code": 400, "message": "环保币不足，快去拍照打卡吧！"}

    # ============= 开始事务操作 =============
    try:
        # 4. 扣除用户【环保币】
        user.eco_coin -= item.points_price

        # 5. 写入积分流水账单 (我们在 models.py 里约定 task_type=4 为商城兑换)
        point_record = models.PointRecord(
            user_id=user.id,
            change_amount=-item.points_price,  # 消费是负数
            task_type=4,
            description=f"商城兑换：{item.name}"
        )
        db.add(point_record)

        # 6. 生成兑换订单 (RedemptionRecord)
        redemption = models.RedemptionRecord(
            user_id=user.id,
            item_id=item.id,
            points_cost=item.points_price,
            status=0  # 默认为0待核销
        )
        db.add(redemption)

        # 7. 扣减真实库存
        if item.stock > 0:
            item.stock -= 1

        db.commit()

        return {
            "code": 200,
            "message": "兑换成功",
            "data": {
                "new_score": user.eco_coin,
                "new_title": user.title
            }
        }
    except Exception as e:
        db.rollback()
        print("兑换发生异常：", e)
        return {"code": 500, "message": "服务器开小差了，请稍后重试"}


# --- 商城兑换记录 删除接口 ---
@app.delete("/api/user/redemption_history/{item_id}")
async def delete_redemption_history(item_id: int, db: Session = Depends(get_db)):
    # 直接物理删除该记录，释放数据库空间
    # (不会影响用户余额，因为发币/扣币的流水在 PointRecord 表中安全保留)
    db.query(models.RedemptionRecord).filter(models.RedemptionRecord.id == item_id).delete(synchronize_session=False)
    db.commit()

    return {"code": 200, "message": "删除成功"}


# 新增获取和生成邀请码的接口

@app.get("/api/admin/invite_codes")
def get_invite_codes(db: Session = Depends(get_db)):
    """获取所有教师邀请码"""
    codes = db.query(models.TeacherInviteCode).order_by(models.TeacherInviteCode.id.desc()).all()
    return {"code": 200, "data": [{"id": c.id, "code": c.code, "is_used": c.is_used} for c in codes]}


@app.post("/api/admin/generate_invite_code")
def generate_invite_code(db: Session = Depends(get_db)):
    """生成一个新的教师邀请码"""
    import random
    import string
    # 生成如 TCH-A1B2C3 的随机码
    random_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    new_code_str = f"TCH-{random_str}"

    new_code = models.TeacherInviteCode(code=new_code_str)
    db.add(new_code)
    db.commit()

    return {"code": 200, "message": "生成成功", "data": {"code": new_code_str}}


# ==========================================
# 接口：获取个人环保成长报告 (家长/学生学情看板)
# ==========================================
# --- 将错题标记为“已消灭(已掌握)” ---
@app.post("/api/user/wrong_book/resolve/{wrong_id}")
async def resolve_wrong_question(wrong_id: int, db: Session = Depends(get_db)):
    # 1. 查找对应的错题记录，且确保它没被逻辑删除
    wrong_item = db.query(models.WrongBook).filter(
        models.WrongBook.id == wrong_id,
        models.WrongBook.is_deleted == False
    ).first()

    if not wrong_item:
        return {"code": 404, "message": "未找到该错题记录"}

    # 2. 将状态更新为 1 (已掌握)
    wrong_item.status = 1
    db.commit()

    return {
        "code": 200,
        "message": "太棒了，这道题你已经掌握啦！",
        "data": {"id": wrong_id, "status": 1}
    }


@app.get("/api/user/growth_report/{user_id}")
async def get_growth_report(user_id: int, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        return {"code": 404, "message": "用户不存在"}

    # ---------------------------------------------------------
    # 维度一：基础信息与击败率 (Beat Percentage)
    # ---------------------------------------------------------
    # 过滤掉 teacher，仅在 student 群体中计算击败率，防止数据干扰
    total_users = db.query(models.User).filter(models.User.role == "student").count()
    lower_score_users = db.query(models.User).filter(
        models.User.role == "student",
        models.User.total_score < user.total_score
    ).count()

    beat_percentage = 0
    if total_users > 1:
        beat_percentage = int((lower_score_users / (total_users - 1)) * 100)
    elif total_users == 1:
        beat_percentage = 100

    # ---------------------------------------------------------
    # 维度二：雷达图数据 (Radar Data) & 寻找薄弱项
    # ---------------------------------------------------------
    category_map = {1: "可回收物", 2: "有害垃圾", 3: "厨余垃圾", 4: "其他垃圾"}
    stats = db.query(models.UserCategoryStat).filter(models.UserCategoryStat.user_id == user_id).all()

    radar_data = []
    highest_category = {"name": "综合", "acc": 0.0}
    lowest_category = {"name": "无", "acc": 1.0}

    stat_dict = {s.category_type: s for s in stats}
    for c_id, c_name in category_map.items():
        if c_id in stat_dict and stat_dict[c_id].total_answered > 0:
            s = stat_dict[c_id]
            acc = round(s.correct_answered / s.total_answered, 2)
        else:
            acc = 0.0

        radar_data.append({
            "category_id": c_id,
            "name": c_name,
            "value": int(acc * 100),  # 新增：供前端直接显示的整数数值
            "accuracy": acc
        })

        if c_id in stat_dict and stat_dict[c_id].total_answered > 0:
            if acc > highest_category["acc"]:
                highest_category = {"name": c_name, "acc": acc}
            if acc < lowest_category["acc"]:
                lowest_category = {"name": c_name, "acc": acc}

    # ---------------------------------------------------------
    # 维度三：近 7 天【三合一】活跃走势 (Activity Trend)
    # ---------------------------------------------------------
    today = datetime.now().date()
    dates_list = [(today - timedelta(days=i)) for i in range(6, -1, -1)]
    dates_str = [d.strftime("%m-%d") for d in dates_list]

    # 建立三维字典容器
    trend = {d: {"recognize": 0, "quiz": 0, "read": 0} for d in dates_str}
    start_date = today - timedelta(days=6)

    recs = db.query(models.RecognizeHistory.created_at).filter(
        models.RecognizeHistory.user_id == user_id,
        models.RecognizeHistory.created_at >= start_date
    ).all()
    quizzes = db.query(models.ChallengeHistory.created_at).filter(
        models.ChallengeHistory.user_id == user_id,
        models.ChallengeHistory.created_at >= start_date
    ).all()
    reads = db.query(models.ReadingRecord.created_at).filter(
        models.ReadingRecord.user_id == user_id,
        models.ReadingRecord.created_at >= start_date
    ).all()

    for r in recs:
        d = r[0].strftime("%m-%d")
        if d in trend: trend[d]["recognize"] += 1
    for q in quizzes:
        d = q[0].strftime("%m-%d")
        if d in trend: trend[d]["quiz"] += 1
    for rd in reads:
        d = rd[0].strftime("%m-%d")
        if d in trend: trend[d]["read"] += 1

    # ---------------------------------------------------------
    # 维度四：学习偏好 (Learning Habit)
    # ---------------------------------------------------------
    recognize_count = db.query(models.RecognizeHistory).filter(models.RecognizeHistory.user_id == user_id).count()
    quiz_count = db.query(models.ChallengeHistory).filter(models.ChallengeHistory.user_id == user_id).count()
    read_count = db.query(models.ReadingRecord).filter(models.ReadingRecord.user_id == user_id).count()

    total_actions = recognize_count + quiz_count + read_count
    preference_tag = "环保小萌新"
    if total_actions > 0:
        if recognize_count / total_actions > 0.5:
            preference_tag = "实践探索家"
        elif quiz_count / total_actions > 0.5:
            preference_tag = "理论小考神"
        elif read_count / total_actions > 0.5:
            preference_tag = "知识博学者"
        else:
            preference_tag = "全能环保卫士"

    # ---------------------------------------------------------
    # 维度五：错题深度分析 (Mistake Clearance & Distribution)
    # ---------------------------------------------------------
    valid_wrong_query = db.query(models.WrongBook).filter(
        models.WrongBook.user_id == user_id,
        models.WrongBook.is_deleted == False
    )

    total_wrong = valid_wrong_query.count()
    cleared_count = valid_wrong_query.filter(models.WrongBook.status == 1).count()

    clear_rate = 0.0
    if total_wrong > 0:
        clear_rate = round(cleared_count / total_wrong, 2)
    elif total_wrong == 0 and quiz_count > 0:
        clear_rate = 1.0

    # 错题分类分布数据
    wrong_distribution = []
    for c_id, c_name in category_map.items():
        if c_id in stat_dict:
            w_count = stat_dict[c_id].total_answered - stat_dict[c_id].correct_answered
            if w_count > 0:
                wrong_distribution.append({"name": c_name, "value": w_count})

    # ---------------------------------------------------------
    # 维度六：AI 智能教育评语引擎
    # ---------------------------------------------------------
    activity_sum = sum(trend[d]["recognize"] + trend[d]["quiz"] + trend[d]["read"] for d in dates_str)

    if total_wrong == 0 and quiz_count > 0:
        ai_comment = "太不可思议了！你的目前没有任何错题积压，是完美的环保学霸！"
    elif activity_sum == 0:
        ai_comment = "最近几天都没有看到你的身影哦，环保习惯需要每天坚持，快来挑战一下吧！"
    elif highest_category["name"] == "综合" and lowest_category["name"] == "无":
        ai_comment = "你还没有积攒足够的答题数据哦，快去【答题闯关】或者【拍照识别】丰富你的档案吧！"
    else:
        ai_comment = f"宝贝是【{highest_category['name']}】小达人！"
        if lowest_category["name"] != highest_category["name"] and lowest_category["acc"] < 0.6:
            ai_comment += f"但在分辨【{lowest_category['name']}】时容易掉进陷阱，属于偏科型选手，建议多复习一下错题本哦！"
        elif clear_rate > 0.8:
            ai_comment += "而且你的错题消化率极高，这种不畏困难的学习态度值得表扬！"
        else:
            ai_comment += "继续保持现在的热情，多去现实里扫一扫垃圾巩固知识吧！"

    # ---------------------------------------------------------
    # 组装返回最终的 JSON 契约
    # ---------------------------------------------------------
    return {
        "code": 200,
        "message": "获取成长报告成功",
        "data": {
            "basic_info": {
                "total_stars": user.total_score,
                "current_title": user.title,
                "beat_percentage": beat_percentage
            },
            "radar_data": radar_data,
            "activity_trend": {
                "dates": dates_str,
                "recognize": [trend[d]["recognize"] for d in dates_str],
                "quiz": [trend[d]["quiz"] for d in dates_str],
                "read": [trend[d]["read"] for d in dates_str]
            },
            "learning_habit": {
                "recognize_count": recognize_count,
                "quiz_count": quiz_count,
                "read_count": read_count,
                "preference_tag": preference_tag
            },
            "mistake_analysis": {
                "cleared_count": cleared_count,
                "total_wrong": total_wrong,
                "clear_rate": clear_rate,
                "distribution": wrong_distribution
            },
            "ai_comment": ai_comment
        }
    }


# ==========================================
# 指导老师端专属 API 模块
# ==========================================
# --- 1. 班级学情大盘 (排行榜与高频错题) ---
# --- 1. 班级学情大盘 (排行榜与高频错题) ---
@app.get("/api/teacher/dashboard")
async def get_teacher_dashboard(teacher_id: int = Query(...), db: Session = Depends(get_db)):
    teacher = db.query(models.User).filter(models.User.id == teacher_id).first()
    target_class_id = teacher.class_id if teacher else 1

    students = db.query(models.User.id, models.User.nickname, models.User.avatar_url, models.User.total_score,
                        models.User.title) \
        .filter(models.User.role == "student", models.User.class_id == target_class_id).all()

    student_ids = [s.id for s in students]

    # 🚀 核心修改：提前建立 UID 到 姓名的映射，供所有图表提取名字！
    uid_to_name = {u.id: u.nickname for u in students}

    # ---------------------------------------------------------
    # 1. 光荣榜
    # ---------------------------------------------------------
    top_students = sorted(students, key=lambda x: x.total_score, reverse=True)[:10]
    leaderboard = [{
        "id": u.id, "nickname": u.nickname, "avatar": u.avatar_url,
        "score": u.total_score, "title": u.title
    } for u in top_students]

    # ---------------------------------------------------------
    # 2. 高频错题 Top 5
    # ---------------------------------------------------------
    top_mistakes = db.query(
        models.WrongBook.item_name, models.WrongBook.correct_answer,
        func.count(func.distinct(models.WrongBook.user_id)).label('err_user_count')
    ).filter(models.WrongBook.user_id.in_(student_ids)) \
        .group_by(models.WrongBook.item_name, models.WrongBook.correct_answer) \
        .order_by(desc('err_user_count')).limit(5).all()

    mistakes_list = [{"item_name": m.item_name, "correct_answer": m.correct_answer, "error_count": m.err_user_count} for
                     m in top_mistakes]

    # ---------------------------------------------------------
    # 3. 分层雷达图
    # ---------------------------------------------------------
    category_map = {1: "可回收物", 2: "有害垃圾", 3: "厨余垃圾", 4: "其他垃圾"}
    all_stats = db.query(models.UserCategoryStat).filter(models.UserCategoryStat.user_id.in_(student_ids)).all()

    category_pass_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    category_active_users = {1: 0, 2: 0, 3: 0, 4: 0}

    for stat in all_stats:
        if stat.total_answered > 0:
            category_active_users[stat.category_type] += 1
            if (stat.correct_answered / stat.total_answered) >= 0.6:
                category_pass_counts[stat.category_type] += 1

    class_radar = []
    for c_id, c_name in category_map.items():
        pass_rate = category_pass_counts[c_id] / category_active_users[c_id] if category_active_users[c_id] > 0 else 0.0
        class_radar.append({"category_id": c_id, "name": c_name, "value": int(pass_rate * 100), "accuracy": pass_rate})

    # ---------------------------------------------------------
    # 4. 双轴活跃走势 (包含 DAU 活跃名单)
    # ---------------------------------------------------------
    today = datetime.now().date()
    dates_list = [(today - timedelta(days=i)) for i in range(6, -1, -1)]
    dates_str = [d.strftime("%m-%d") for d in dates_list]
    start_date = today - timedelta(days=6)

    trend = {d: {"recognize": 0, "quiz": 0, "read": 0, "active_users": set()} for d in dates_str}

    recs = db.query(models.RecognizeHistory.created_at, models.RecognizeHistory.user_id).filter(
        models.RecognizeHistory.user_id.in_(student_ids), models.RecognizeHistory.created_at >= start_date).all()
    quizzes = db.query(models.ChallengeHistory.created_at, models.ChallengeHistory.user_id).filter(
        models.ChallengeHistory.user_id.in_(student_ids), models.ChallengeHistory.created_at >= start_date).all()
    reads = db.query(models.ReadingRecord.created_at, models.ReadingRecord.user_id).filter(
        models.ReadingRecord.user_id.in_(student_ids), models.ReadingRecord.created_at >= start_date).all()

    for r in recs:
        d = r[0].strftime("%m-%d")
        if d in trend: trend[d]["recognize"] += 1; trend[d]["active_users"].add(r[1])
    for q in quizzes:
        d = q[0].strftime("%m-%d")
        if d in trend: trend[d]["quiz"] += 1; trend[d]["active_users"].add(q[1])
    for rd in reads:
        d = rd[0].strftime("%m-%d")
        if d in trend: trend[d]["read"] += 1; trend[d]["active_users"].add(rd[1])

    # 🚀 核心修改：打包 DAU 具体名单给前端
    dau_names = []
    for d in dates_str:
        names = [uid_to_name.get(uid, f"学生{uid}") for uid in trend[d]["active_users"]]
        dau_names.append(names)

    class_activity = {
        "dates": dates_str,
        "recognize": [trend[d]["recognize"] for d in dates_str],
        "quiz": [trend[d]["quiz"] for d in dates_str],
        "read": [trend[d]["read"] for d in dates_str],
        "dau": [len(trend[d]["active_users"]) for d in dates_str],
        "dau_names": dau_names  # 👈 传给前端的名单
    }

    # ---------------------------------------------------------
    # 5. 班级学习基因人群分布 (名单)
    # ---------------------------------------------------------
    user_actions = {uid: {"rec": 0, "quiz": 0, "read": 0} for uid in student_ids}

    rec_c = db.query(models.RecognizeHistory.user_id, func.count(models.RecognizeHistory.id)).filter(
        models.RecognizeHistory.user_id.in_(student_ids)).group_by(models.RecognizeHistory.user_id).all()
    quiz_c = db.query(models.ChallengeHistory.user_id, func.count(models.ChallengeHistory.id)).filter(
        models.ChallengeHistory.user_id.in_(student_ids)).group_by(models.ChallengeHistory.user_id).all()
    read_c = db.query(models.ReadingRecord.user_id, func.count(models.ReadingRecord.id)).filter(
        models.ReadingRecord.user_id.in_(student_ids)).group_by(models.ReadingRecord.user_id).all()

    for uid, c in rec_c: user_actions[uid]["rec"] = c
    for uid, c in quiz_c: user_actions[uid]["quiz"] = c
    for uid, c in read_c: user_actions[uid]["read"] = c

    # 🚀 核心修改：改为存放名单列表
    habit_groups = {"实践探索派": [], "理论小考神": [], "知识博学者": [], "全能卫士": [], "暂无数据": []}

    for uid, actions in user_actions.items():
        tot = actions["rec"] + actions["quiz"] + actions["read"]
        nickname = uid_to_name.get(uid, f"学生{uid}")
        if tot == 0:
            habit_groups["暂无数据"].append(nickname)
        elif actions["rec"] / tot > 0.5:
            habit_groups["实践探索派"].append(nickname)
        elif actions["quiz"] / tot > 0.5:
            habit_groups["理论小考神"].append(nickname)
        elif actions["read"] / tot > 0.5:
            habit_groups["知识博学者"].append(nickname)
        else:
            habit_groups["全能卫士"].append(nickname)

    habit_pie = [{"name": k, "value": len(v), "students": v} for k, v in habit_groups.items() if len(v) > 0]

    # ---------------------------------------------------------
    # 6. 错题歼灭战况分布图 (名单)
    # ---------------------------------------------------------
    wrong_stats = db.query(
        models.WrongBook.user_id, func.count(models.WrongBook.id).label("total"),
        func.sum(models.WrongBook.status).label("cleared")
    ).filter(models.WrongBook.user_id.in_(student_ids), models.WrongBook.is_deleted == False).group_by(
        models.WrongBook.user_id).all()

    # 🚀 核心修改：改为存放名单列表
    clearance_buckets = {"摆烂区\n(0-20%)": [], "拖延区\n(21-59%)": [], "良好区\n(60-89%)": [], "清零区\n(≥90%)": []}

    total_class_wrong = 0
    total_class_cleared = 0

    for stat in wrong_stats:
        uid, tot, cleared = stat
        cleared = int(cleared) if cleared else 0
        total_class_wrong += tot
        total_class_cleared += cleared

        name = uid_to_name.get(uid, f"学生{uid}")

        if tot > 0:
            rate = cleared / tot
            if rate <= 0.2:
                clearance_buckets["摆烂区\n(0-20%)"].append(name)
            elif rate < 0.6:
                clearance_buckets["拖延区\n(21-59%)"].append(name)
            elif rate < 0.9:
                clearance_buckets["良好区\n(60-89%)"].append(name)
            else:
                clearance_buckets["清零区\n(≥90%)"].append(name)

    clearance_bar = {
        "categories": list(clearance_buckets.keys()),
        "values": [len(v) for v in clearance_buckets.values()],
        "students": list(clearance_buckets.values()),  # 👈 传给前端的名单
        "avg_rate": round(total_class_cleared / total_class_wrong, 2) if total_class_wrong > 0 else 1.0
    }

    return {
        "code": 200, "message": "获取大盘数据成功",
        "data": {
            "leaderboard": leaderboard,
            "top_mistakes": mistakes_list,
            "class_radar": class_radar,
            "class_activity": class_activity,
            "habit_pie": habit_pie,
            "clearance_dist": clearance_bar
        }
    }

# --- 2. 获取待核销(待发奖)的订单列表 ---
@app.get("/api/teacher/pending_orders")
async def get_pending_orders(teacher_id: int = Query(...), db: Session = Depends(get_db)):
    teacher = db.query(models.User).filter(models.User.id == teacher_id).first()
    target_class_id = teacher.class_id if teacher else 1

    # 只查 user.class_id 等于老师 class_id 的订单
    orders = db.query(models.RedemptionRecord).join(
        models.User, models.RedemptionRecord.user_id == models.User.id
    ).filter(
        models.RedemptionRecord.status == 0,
        models.User.class_id == target_class_id  # 隔离过滤
    ).order_by(models.RedemptionRecord.created_at.asc()).all()

    res_list = []
    for order in orders:
        student = db.query(models.User).filter(models.User.id == order.user_id).first()
        item = db.query(models.MallItem).filter(models.MallItem.id == order.item_id).first()

        if student and item:
            res_list.append({
                "order_id": order.id,
                "student_id": student.id,
                "student_name": student.nickname,
                "student_avatar": student.avatar_url,
                "item_name": item.name,
                "item_image": item.image_url,
                "cost": order.points_cost,
                "created_at": order.created_at.strftime("%m-%d %H:%M")
            })

    return {"code": 200, "message": "获取待核销列表成功", "data": res_list}


# --- 3. 老师执行核销动作 (发奖) ---
from pydantic import BaseModel


class VerifyOrderReq(BaseModel):
    order_id: int
    teacher_id: int


@app.post("/api/teacher/verify")
async def verify_student_order(req: VerifyOrderReq, db: Session = Depends(get_db)):
    order = db.query(models.RedemptionRecord).filter(models.RedemptionRecord.id == req.order_id).first()

    if not order:
        return {"code": 404, "message": "找不到该订单"}
    if order.status == 1:
        return {"code": 400, "message": "该奖品已经被核销过了，请勿重复操作"}

    # 确认核销：更新状态，并打上当前操作老师的思想钢印和时间戳
    order.status = 1
    order.verified_by = req.teacher_id
    order.verified_at = datetime.now()

    db.commit()

    return {"code": 200, "message": "🎉 核销成功！奖品已发放"}


# ==========================================
# 通用图片上传接口 (复用腾讯云 COS)
# ==========================================
@app.post("/api/upload")
async def upload_common_image(file: UploadFile = File(...)):
    try:
        # 1. 读取文件二进制内容
        file_content = await file.read()

        # 2. 生成一个永不重复的文件名，并放在 mall_items 文件夹下
        file_extension = file.filename.split(".")[-1] if "." in file.filename else "jpg"
        unique_filename = f"mall_items/{uuid.uuid4().hex}.{file_extension}"

        # 3. 调用你之前封装好的 COS 上传函数
        image_url = upload_file_to_cos(file_content, unique_filename)

        if not image_url:
            return {"code": 500, "message": "图片上传到云端失败"}

        # 4. 完美返回给前端
        return {
            "code": 200,
            "message": "上传成功",
            "data": {"url": image_url}
        }
    except Exception as e:
        return {"code": 500, "message": f"服务器异常: {str(e)}"}

# --- 老师发布新奖品 ---
@app.post("/api/teacher/mall/add")
async def add_mall_item(item_data: schemas.MallItemCreate, db: Session = Depends(get_db)):
    teacher = db.query(models.User).filter(models.User.id == item_data.teacher_id).first()

    new_item = models.MallItem(
        name=item_data.name,
        desc=item_data.desc,
        points_price=item_data.points_price,
        image_url=item_data.image_url,
        stock=item_data.stock,
        created_by=item_data.teacher_id,
        class_id=teacher.class_id  # 商品归属到老师的班级
    )
    db.add(new_item)
    db.commit()
    return {"code": 200, "message": "奖品发布成功！"}


# --- 老师管理奖品列表（包含下架功能） ---
@app.get("/api/teacher/mall/list")
async def get_teacher_mall_items(
        teacher_id: int = Query(..., description="当前操作的老师ID"),
        db: Session = Depends(get_db)
):
    # 2. 获取该老师所在的班级
    teacher = db.query(models.User).filter(models.User.id == teacher_id).first()
    target_class_id = teacher.class_id if teacher else 1

    # 3. 🚀 核心修改：只查询所属班级是该老师班级，并且【未被逻辑删除】的奖品
    items = db.query(models.MallItem).filter(
        models.MallItem.class_id == target_class_id,
        models.MallItem.is_deleted == False  # 👈 关键点：不在库房中显示已删除的商品
    ).order_by(models.MallItem.created_at.desc()).all()

    return {"code": 200, "data": items}


# --- 切换奖品状态（上架/下架） ---
@app.post("/api/teacher/mall/toggle/{item_id}")
async def toggle_mall_item(item_id: int, db: Session = Depends(get_db)):
    item = db.query(models.MallItem).filter(models.MallItem.id == item_id).first()
    if not item: return {"code": 404, "message": "商品不存在"}

    item.is_active = not item.is_active
    db.commit()
    return {"code": 200, "message": "操作成功", "new_status": item.is_active}


# ==========================================
# 物理删除已下架的商品 (带未核销保护锁)
# ==========================================
@app.delete("/api/teacher/mall/items/{item_id}")
async def delete_off_shelf_item(item_id: int, teacher_id: int = Query(...), db: Session = Depends(get_db)):
    # 1. 权限校验
    user = db.query(models.User).filter(models.User.id == teacher_id).first()
    if not user or user.role != "teacher":
        return {"code": 403, "message": "权限不足"}

    # 2. 查找商品
    item = db.query(models.MallItem).filter(models.MallItem.id == item_id).first()
    if not item or item.is_deleted:
        return {"code": 404, "message": "找不到该商品或已被删除"}

    # 3. 必须先下架
    if item.is_active:
        return {
            "code": 400,
            "message": "删除失败！当前商品处于【上架中】状态，请先下架。"
        }

    # 4. 🚀 核心精准修复：使用真实的 RedemptionRecord 表，且 status == 0 代表待核销
    pending_count = db.query(models.RedemptionRecord).filter(
        models.RedemptionRecord.item_id == item_id,
        models.RedemptionRecord.status == 0  # 0:待核销(未发货)
    ).count()

    if pending_count > 0:
        return {
            "code": 400,
            "message": f"删除拦截：还有 {pending_count} 名学生兑换了该商品但【未核销】，请先完成核销发放后再删除！"
        }

    # 5. 执行逻辑删除，而非物理删除
    try:
        item.is_deleted = True  # 打上删除标记，数据依然留在数据库保护历史记录
        db.commit()
        return {"code": 200, "message": "商品已从库房永久移除"}
    except Exception as e:
        db.rollback()
        return {"code": 500, "message": f"操作失败: {str(e)}"}

# ==========================================
# 通用接口：获取 年级-班级 级联字典树
# ==========================================
@app.get("/api/common/class_options")
async def get_class_options(db: Session = Depends(get_db)):
    classes = db.query(models.SchoolClass).all()

    # 将扁平数据按 grade_name 分组组合成树状结构
    grade_dict = {}
    for c in classes:
        # 跳过系统默认的隐藏班级（ID为1的系统测试班，不给普通用户选）
        if c.id == 1:
            continue

        if c.grade_name not in grade_dict:
            grade_dict[c.grade_name] = []

        grade_dict[c.grade_name].append({
            "id": c.id,
            "name": c.class_name
        })

    # 格式化为前端 Picker 容易解析的数组
    result = []
    for grade, cls_list in grade_dict.items():
        result.append({
            "grade_name": grade,
            "classes": cls_list
        })

    return {"code": 200, "message": "获取成功", "data": result}


# ==========================================
# 首页通知系统新增接口
# ==========================================
from sqlalchemy import or_, desc
# 1. 获取用户的未读通知
@app.get("/api/user/notifications")
async def get_notifications(user_id: int, db: Session = Depends(get_db)):
    now = datetime.now()

    # 🚀 核心查询：找该用户的、未读的、且（没过期 或者 永久有效）的通知
    notices = db.query(models.Notification).filter(
        models.Notification.user_id == user_id,
        models.Notification.is_read == False,
        or_(models.Notification.expires_at == None, models.Notification.expires_at > now)
    ).order_by(desc(models.Notification.created_at)).all()

    # 打包返回，顺便把时间格式化为 '04-14 10:30' 的友好格式供前端展示
    result = [{
        "id": n.id,
        "type": n.type,
        "content": n.content,
        "created_at": n.created_at.strftime("%m-%d %H:%M")
    } for n in notices]

    return {"code": 200, "data": result}

# 2. 将通知标为已读
@app.post("/api/user/notifications/{notice_id}/read")
async def read_notification(notice_id: int, db: Session = Depends(get_db)):
    notice = db.query(models.Notification).filter(models.Notification.id == notice_id).first()
    if notice:
        notice.is_read = True
        db.commit()
    return {"code": 200, "message": "已标为已读"}


# ==========================================
# 接口：获取班级学生名单 (供老师发通知下拉框使用)
# ==========================================
@app.get("/api/teacher/students")
async def get_teacher_students(teacher_id: int, db: Session = Depends(get_db)):
    teacher = db.query(models.User).filter(models.User.id == teacher_id).first()
    target_class_id = teacher.class_id if teacher else 1

    students = db.query(models.User.id, models.User.nickname).filter(
        models.User.role == "student",
        models.User.class_id == target_class_id
    ).all()

    # 第一项固定塞入群发选项
    data = [{"id": 0, "name": "📢 广播：全体学生"}] + [{"id": s.id, "name": f"👤 {s.nickname}"} for s in students]
    return {"code": 200, "data": data}


# ==========================================
# 接口：老师发送通知 (支持群发和单发)
# ==========================================
class SendNoticeSchema(BaseModel):
    user_id: int    # 0 代表发给全体，非 0 代表单个学生
    type: str       # 通知类型 (如: '日常提醒', '任务布置', '奖励通报')
    duration: int   # 生效时长(单位:小时)，传 0 代表永久
    content: str    # 通知内容


@app.post("/api/teacher/send_notice")
async def send_teacher_notice(req: SendNoticeSchema, teacher_id: int = Query(...), db: Session = Depends(get_db)):
    # 🚀 1. 计算过期时间
    expire_time = None
    if req.duration > 0:
        expire_time = datetime.now() + timedelta(hours=req.duration)

    if req.user_id == 0:
        # 🚀 2. 群发逻辑：找出老师对应班级的所有学生
        teacher = db.query(models.User).filter(models.User.id == teacher_id).first()
        target_class_id = teacher.class_id if teacher else 1
        students = db.query(models.User.id).filter(
            models.User.role == "student",
            models.User.class_id == target_class_id
        ).all()

        notices = [
            models.Notification(
                user_id=s.id,
                type=req.type,
                content=req.content,
                expires_at=expire_time
            ) for s in students
        ]
        db.bulk_save_objects(notices)
    else:
        # 🚀 3. 单发逻辑
        new_notice = models.Notification(
            user_id=req.user_id,
            type=req.type,
            content=req.content,
            expires_at=expire_time
        )
        db.add(new_notice)

    db.commit()
    return {"code": 200, "message": "通知已送达学生"}


# 4. 手动触发周榜奖励 (方便你答辩演示)
@app.post("/api/teacher/trigger_weekly_reward")
async def trigger_reward():
    run_weekly_settlement()
    return {"code": 200, "message": "结算完成，奖励与通知已下发"}