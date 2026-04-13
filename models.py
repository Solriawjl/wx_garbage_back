# 将SQL表结构“翻译”成Python面向对象的代码（ORM）
from sqlalchemy import Column, Integer, String, DateTime, Numeric, Text, Enum, ForeignKey, Boolean
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from database import Base
import enum

# --- 枚举类型定义 ---
class FeedbackType(str, enum.Enum):
    image = "image"
    text = "text"

class ConfigType(str, enum.Enum):
    banner = "banner"
    daily_tip = "daily_tip"

# --- 0. 班级架构表 (新增) ---
class SchoolClass(Base):
    """
    班级行政架构表，用于实现多班级数据域控隔离
    """
    __tablename__ = "school_classes"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    grade_name = Column(String(50), nullable=False, comment="年级名称，如：一年级")
    class_name = Column(String(50), nullable=False, comment="班级名称，如：1班")
    created_at = Column(DateTime, server_default=func.now())

    # 方便反向查询：获取这个班级下的所有学生和奖品
    users = relationship("User", back_populates="school_class")
    mall_items = relationship("MallItem", back_populates="school_class")

# --- 1. 用户表 ---
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True, comment="主键ID")
    openid = Column(String(100), unique=True, nullable=False, index=True, comment="微信用户唯一标识")
    role = Column(String(20), default="student", comment="角色: student/teacher")
    nickname = Column(String(100), default="微信用户", comment="用户昵称")
    avatar_url = Column(String(255), nullable=True, comment="用户头像链接")
    total_score = Column(Integer, default=0, comment="答题总积分")
    eco_coin = Column(Integer, default=0, comment="环保币(用于商城消费)")
    title = Column(String(50), default="环保新手", comment="当前环保称号")
    created_at = Column(DateTime, server_default=func.now(), comment="注册时间")
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), comment="最后更新时间")
    # 关联到具体班级 (这里 nullable=True 允许新用户注册时先为空，再选择)
    class_id = Column(Integer, ForeignKey("school_classes.id"), nullable=True, comment="所属班级ID")

    # 关联属性，方便代码里直接用 user.school_class.grade_name
    school_class = relationship("SchoolClass", back_populates="users")
    # 关联属性：方便通过 user.recognize_histories 直接获取该用户的所有历史记录
    recognize_histories = relationship("RecognizeHistory", back_populates="user")
    challenge_histories = relationship("ChallengeHistory", back_populates="user")
    wrong_books = relationship("WrongBook", back_populates="user")
    feedbacks = relationship("Feedback", back_populates="user")


# --- 教师邀请码表 ---
class TeacherInviteCode(Base):
    __tablename__ = "teacher_invite_codes"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    code = Column(String(20), unique=True, nullable=False, index=True, comment="邀请码")
    is_used = Column(Boolean, default=False, comment="是否已被使用")
    used_by = Column(Integer, ForeignKey("users.id"), nullable=True, comment="使用者ID")
    created_at = Column(DateTime, server_default=func.now(), comment="生成时间")

# --- 2. 垃圾分类字典表 ---
class GarbageItem(Base):
    __tablename__ = "garbage_items"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    item_name = Column(String(100), nullable=False, index=True, comment="物品名称")
    category_type = Column(Integer, nullable=False, comment="分类：1-可回收, 2-有害, 3-厨余, 4-其他")
    image_url = Column(String(255), nullable=True, comment="物品示意图链接") # 👈 新增这一行
    tips = Column(String(255), nullable=True, comment="投放提示")
    created_at = Column(DateTime, server_default=func.now())
    sub_category = Column(String(50), default="其他类")

# --- 3. 识别历史记录表 ---
class RecognizeHistory(Base):
    __tablename__ = "recognize_history"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    image_url = Column(String(255), nullable=False, comment="COS图片链接")
    recognized_name = Column(String(100), nullable=False, comment="AI识别名称")
    category_type = Column(Integer, nullable=False, comment="分类类型")
    confidence = Column(Numeric(5, 2), nullable=True, comment="置信度")
    created_at = Column(DateTime, server_default=func.now(), comment="识别时间")
    is_deleted = Column(Boolean, default=False, comment="逻辑删除状态: False-可见, True-用户已删除")

    # 反向关联
    user = relationship("User", back_populates="recognize_histories")


# --- 4. 挑战记录表 ---
class ChallengeHistory(Base):
    __tablename__ = "challenge_history"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    score = Column(Integer, nullable=False, comment="得分")
    correct_count = Column(Integer, nullable=False, comment="答对题数")
    created_at = Column(DateTime, server_default=func.now(), comment="答题时间")
    is_deleted = Column(Boolean, default=False, comment="逻辑删除状态: False-可见, True-用户已删除")
    mode = Column(String(50), default="classic")  # 'classic' 经典模式 / 'timed' 计时模式
    total_count = Column(Integer, default=10)  # 该局总共做了多少题
    user = relationship("User", back_populates="challenge_histories")


# --- 5. 错题本表 ---
class WrongBook(Base):
    __tablename__ = "wrong_book"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    item_name = Column(String(100), nullable=False, comment="做错的题目")
    user_answer = Column(String(50), nullable=False, comment="用户的错误选项")
    correct_answer = Column(String(50), nullable=False, comment="正确答案")
    status = Column(Integer, default=0, comment="0-待复习(未掌握), 1-已消灭(已掌握)")
    created_at = Column(DateTime, server_default=func.now(), comment="做错时间")
    is_deleted = Column(Boolean, default=False, comment="逻辑删除状态: False-可见, True-用户已删除")

    user = relationship("User", back_populates="wrong_books")


# --- 6. 纠错反馈表 ---
class Feedback(Base):
    __tablename__ = "feedbacks"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    type = Column(Enum(FeedbackType), nullable=False, comment="image/text")
    image_url = Column(String(255), nullable=True, comment="照片链接")
    item_name = Column(String(100), nullable=False, comment="原搜索词/AI误判结果")
    suggestion = Column(String(100), nullable=False, comment="用户建议分类")
    status = Column(Integer, default=0, comment="0-待处理, 1-已采纳, 2-已驳回")
    admin_reply = Column(Text, nullable=True, comment="管理员回复")
    created_at = Column(DateTime, server_default=func.now(), comment="提交时间")
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), comment="处理时间")
    is_deleted = Column(Boolean, default=False, comment="逻辑删除状态: False-可见, True-用户已删除")

    user = relationship("User", back_populates="feedbacks")


# --- 7. 首页配置表 (可选) ---
class HomeConfig(Base):
    __tablename__ = "home_configs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    type = Column(Enum(ConfigType), nullable=False, comment="banner/daily_tip")
    content = Column(String(255), nullable=False, comment="图片链接/物品名")
    is_active = Column(Boolean, default=True, comment="是否启用")
    sort_order = Column(Integer, default=0, comment="排序权重")


# --- 8. 四大垃圾分类详情表 ---
class GarbageCategory(Base):
    __tablename__ = "garbage_categories"

    id = Column(Integer, primary_key=True, index=True, comment="分类ID：1-可回收, 2-有害, 3-厨余, 4-其他")
    category_name = Column(String(20), nullable=False, comment="中文名称")
    category_class = Column(String(20), nullable=False, comment="前端CSS类名")

    # --- 原有字段 ---
    eco_value = Column(Text, nullable=False, comment="环保价值")
    put_guidance = Column(Text, nullable=False, comment="通用一句话投放指导")

    # --- 教育闭环与日式严谨标准字段 ---
    harm_description = Column(Text, nullable=True, comment="如果不分类的危害（儿童科普语气）")
    process_method = Column(Text, nullable=True, comment="回收/处理的生命周期（它最后变成了什么）")
    sub_guidance = Column(Text, nullable=True, comment="各个官方小类的投放前置动作指导")

# --- 9. 环保科普小知识表 ---
class EnvironmentalTip(Base):
    __tablename__ = "environmental_tips"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(100), nullable=False, comment="标题")
    content = Column(Text, nullable=False, comment="内容")
    image_url = Column(String(255), nullable=True, comment="配图")
    view_count = Column(Integer, default=0, comment="阅读量")
    created_at = Column(DateTime, server_default=func.now(), comment="发布时间")

# --- 10. 低置信度难例收集表 (Auto-collected Hard Examples) ---
class LowConfidenceRecord(Base):
    __tablename__ = "low_confidence_records"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    image_url = Column(String(255), nullable=False, comment="COS原始图片链接")
    ai_predicted_category = Column(Integer, nullable=False, comment="当时AI预测的大类ID")
    confidence = Column(Numeric(5, 2), nullable=False, comment="当时的低置信度")
    status = Column(Integer, default=0, comment="0-待标注, 1-已打标入库, 2-废弃(如图片模糊)")
    created_at = Column(DateTime, server_default=func.now(), comment="收集时间")


class PointRecord(Base):
    """
    积分流水表：记录用户积分的获取与消耗轨迹
    """
    __tablename__ = "point_records"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    # 变动数量：获取为正数（如 +10），消耗为负数（如 -500）
    change_amount = Column(Integer, nullable=False, comment="积分变动数量")

    # 任务类型，用于后端防刷逻辑判定：
    # 1: 每日首次拍照打卡
    # 2: 每日阅读科普知识
    # 3: 环保挑战得分
    # 4: 商城积分兑换消耗
    task_type = Column(Integer, nullable=False, comment="任务或行为类型")

    # 具体的中文描述，方便前端直接展示给用户看
    description = Column(String(100), nullable=False, comment="账单描述")

    created_at = Column(DateTime, server_default=func.now(), comment="记录时间")


# ==========================================
# 积分商城模块 数据模型
# ==========================================

class MallItem(Base):
    """
    商城商品表
    """
    __tablename__ = "mall_items"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(100), nullable=False, comment="商品名称")
    desc = Column(String(200), comment="商品描述")
    points_price = Column(Integer, nullable=False, comment="兑换所需积分")
    image_url = Column(String(255), comment="商品图片")
    # 将奖品与班级绑定，实现商城商品隔离
    class_id = Column(Integer, ForeignKey("school_classes.id"), nullable=True, comment="所属班级ID(为空代表全校通用)")

    # 反向关联
    school_class = relationship("SchoolClass", back_populates="mall_items")
    # 库存设计：-1代表无限库存（如虚拟勋章），大于0代表实体商品真实库存
    stock = Column(Integer, default=-1, comment="库存数量")

    is_active = Column(Boolean, default=True, comment="是否上架")
    created_by = Column(Integer, ForeignKey("users.id"), nullable=True, comment="发布者ID")
    created_at = Column(DateTime, server_default=func.now())
    creator = relationship("User")

class RedemptionRecord(Base):
    """
    商品兑换记录（订单表）
    """
    __tablename__ = "redemption_records"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    item_id = Column(Integer, ForeignKey("mall_items.id"), nullable=False)

    points_cost = Column(Integer, nullable=False, comment="当时花费的积分")

    # 状态：0-待核销(未发货), 1-已核销(已完成)
    status = Column(Integer, default=0, comment="核销状态")
    verified_by = Column(Integer, ForeignKey("users.id"), nullable=True, comment="核销老师的ID")
    verified_at = Column(DateTime, nullable=True, comment="核销时间")
    created_at = Column(DateTime, server_default=func.now())


# --- 11. 用户学情雷达统计表 (新增) ---
class UserCategoryStat(Base):
    """
    记录用户在四大分类上的答题正确率，用于绘制雷达图
    """
    __tablename__ = "user_category_stats"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    # 类别：1-可回收, 2-有害, 3-厨余, 4-其他
    category_type = Column(Integer, nullable=False, index=True)

    # 统计数据
    total_answered = Column(Integer, default=0, comment="该类别总共答了多少次")
    correct_answered = Column(Integer, default=0, comment="该类别答对了多少次")

    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    user = relationship("User")


# --- 12. 环保知识阅读记录表 (新增) ---
class ReadingRecord(Base):
    """
    记录用户点击查看环保知识的行为，用于统计活跃度和学习偏好
    """
    __tablename__ = "reading_records"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    # 关联到你 SystemConfig 表里的某条知识 id
    tip_id = Column(Integer, comment="阅读的知识条目ID")

    created_at = Column(DateTime, server_default=func.now(), comment="阅读时间")

    user = relationship("User")