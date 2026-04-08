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

class CategoryStatItem(BaseModel):
    category_type: int  # 分类ID：1-可回收, 2-有害, 3-厨余, 4-其他
    total: int          # 本次挑战遇到该分类的题目总数
    correct: int        # 答对的数量

class QuizSubmitRequest(BaseModel):
    user_id: int
    score: int            # 本次得分 (比如答对1题得10分)
    correct_count: int    # 答对题数
    wrong_answers: List[QuizWrongAnswer] = [] # 错题数组，全对就是空数组
    mode: Optional[str] = "classic"
    total_count: Optional[int] = 10
    time_used: Optional[int] = 0  # 用户实际耗时(秒)
    total_time: Optional[int] = 60  # 该局总时长(秒)
    # 接收前端传来的分类埋点数据
    category_stats: Optional[List[CategoryStatItem]] = []

# ==========================================
# 9. 纠错反馈提交 (Feedback Submit) Schema
# ==========================================
class FeedbackSubmitRequest(BaseModel):
    user_id: int
    type: str                  # 传 'image' 或 'text'
    image_url: Optional[str] = None # 照片链接，如果是文字搜索纠错则为空
    item_name: str             # 原来的错误结果 (比如AI说它是"可回收物")
    suggestion: str            # 用户选择的正确分类 (如果填了具体物品，就拼在后面)

# ==========================================
# 10. 后台管理系统 (Admin Web) 专用 Schema
# ==========================================
class AdminLoginRequest(BaseModel):
    username: str = Field(..., description="管理员账号")
    password: str = Field(..., description="管理员密码")

# ==========================================
# 10. 家长端/学生端 学情成长报告 (Growth Report) Schema
# ==========================================
from pydantic import BaseModel
from typing import List

class ReportBasicInfo(BaseModel):
    total_stars: int            # 累计环保星
    current_title: str          # 当前称号
    beat_percentage: int        # 击败了百分之多少的同学

class RadarDataItem(BaseModel):
    category_id: int            # 1, 2, 3, 4
    name: str                   # "可回收物" 等
    accuracy: float             # 正确率 0.0 ~ 1.0

class ActivityTrend(BaseModel):
    dates: List[str]            # 近7天的日期数组，如 ["04-02", "04-03", ...]
    counts: List[int]           # 对应的每日活跃次数，如 [2, 0, 1, ...]

class LearningHabit(BaseModel):
    recognize_count: int        # 累计拍照识别次数 (实践)
    quiz_count: int             # 累计答题闯关次数 (理论)
    preference_tag: str         # 偏好标签："实践探索家" / "理论小考神" / "全能环保卫士"

class MistakeClearance(BaseModel):
    cleared_count: int          # 已消灭的错题数 (status=1)
    total_wrong: int            # 历史总错题数
    clear_rate: float           # 消化率 0.0 ~ 1.0

# 👇 这个是最终发给前端的顶级总结构
class GrowthReportResponse(BaseModel):
    basic_info: ReportBasicInfo
    radar_data: List[RadarDataItem]
    activity_7_days: ActivityTrend
    learning_habit: LearningHabit
    mistake_clearance: MistakeClearance
    ai_comment: str             # 智能生成的总体评语

# --- 11. 教师端奖品管理 Schema ---
class MallItemCreate(BaseModel):
    name: str
    desc: Optional[str] = None
    points_price: int
    image_url: str
    stock: int = -1
    teacher_id: int # 发布老师的ID

# --- 12. 订单核销请求 ---
class OrderVerifyRequest(BaseModel):
    order_id: int
    teacher_id: int