# 对前端请求进行数据校验
# 对后端数据转换为JSON格式
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum

# 自动转化为 JSON
class ORMBaseConfig:
    from_attributes = True


# ==========================================
# 1. 微信登录授权专用 Schema
# ==========================================
class WxLoginRequest(BaseModel):
    code: str = Field(..., description="前端调用 wx.login 获取的临时凭证")


# ==========================================
# 2. 用户 (User) Schema
# ==========================================
class UserBase(BaseModel):
    nickname: Optional[str] = "微信用户"
    avatar_url: Optional[str] = None

class UserResponse(UserBase):
    id: int
    openid: str
    total_score: int
    title: str
    created_at: datetime

    class Config(ORMBaseConfig):
        pass


# ==========================================
# 3. 识别历史 (RecognizeHistory) Schema
# ==========================================
class RecognizeHistoryCreate(BaseModel):
    image_url: str
    recognized_name: str
    category_type: int
    confidence: Optional[float] = None

class RecognizeHistoryResponse(RecognizeHistoryCreate):
    id: int
    created_at: datetime

    class Config(ORMBaseConfig):
        pass


# ==========================================
# 4. 挑战历史 (ChallengeHistory) Schema
# ==========================================
class ChallengeHistoryCreate(BaseModel):
    score: int
    correct_count: int
    earned_title: str

class ChallengeHistoryResponse(ChallengeHistoryCreate):
    id: int
    created_at: datetime

    class Config(ORMBaseConfig):
        pass


# ==========================================
# 5. 错题本 (WrongBook) Schema
# ==========================================
class WrongBookCreate(BaseModel):
    item_name: str
    user_answer: str
    correct_answer: str

class WrongBookResponse(WrongBookCreate):
    id: int
    created_at: datetime

    class Config(ORMBaseConfig):
        pass


# ==========================================
# 6. 纠错反馈 (Feedback) Schema
# ==========================================
class FeedbackTypeEnum(str, Enum):
    image = "image"
    text = "text"

class FeedbackCreate(BaseModel):
    type: FeedbackTypeEnum
    image_url: Optional[str] = None
    item_name: str
    suggestion: str

class FeedbackResponse(FeedbackCreate):
    id: int
    status: int
    admin_reply: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config(ORMBaseConfig):
        pass


# ==========================================
# 7. 垃圾分类字典 (GarbageItem) Schema
# ==========================================
class GarbageItemResponse(BaseModel):
    id: int
    item_name: str
    category_type: int
    tips: Optional[str] = None

    class Config(ORMBaseConfig):
        pass

# ==========================================
# 8. 挑战答题交卷 (Quiz Submit) Schema
# ==========================================
class QuizWrongAnswer(BaseModel):
    item_name: str
    user_answer: str      # 用户选错的类别名 (比如 "可回收物")
    correct_answer: str   # 正确的类别名 (比如 "厨余垃圾")

class QuizSubmitRequest(BaseModel):
    user_id: int
    score: int            # 本次得分 (比如答对1题得10分)
    correct_count: int    # 答对题数
    wrong_answers: List[QuizWrongAnswer] = [] # 错题数组，全对就是空数组

# ==========================================
# 9. 纠错反馈提交 (Feedback Submit) Schema
# ==========================================
class FeedbackSubmitRequest(BaseModel):
    user_id: int
    type: str                  # 传 'image' 或 'text'
    image_url: Optional[str] = None # 照片链接，如果是文字搜索纠错则为空
    item_name: str             # 原来的错误结果 (比如AI说它是"可回收物")
    suggestion: str            # 用户选择的正确分类 (如果填了具体物品，就拼在后面)