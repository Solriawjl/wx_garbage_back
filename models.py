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


# --- 1. 用户表 ---
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True, comment="主键ID")
    openid = Column(String(100), unique=True, nullable=False, index=True, comment="微信用户唯一标识")
    nickname = Column(String(100), default="微信用户", comment="用户昵称")
    avatar_url = Column(String(255), nullable=True, comment="用户头像链接")
    total_score = Column(Integer, default=0, comment="答题总积分")
    title = Column(String(50), default="环保新手", comment="当前环保称号")
    created_at = Column(DateTime, server_default=func.now(), comment="注册时间")
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), comment="最后更新时间")

    # 关联属性：方便通过 user.recognize_histories 直接获取该用户的所有历史记录
    recognize_histories = relationship("RecognizeHistory", back_populates="user")
    challenge_histories = relationship("ChallengeHistory", back_populates="user")
    wrong_books = relationship("WrongBook", back_populates="user")
    feedbacks = relationship("Feedback", back_populates="user")


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

    # 反向关联
    user = relationship("User", back_populates="recognize_histories")


# --- 4. 挑战记录表 ---
class ChallengeHistory(Base):
    __tablename__ = "challenge_history"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    score = Column(Integer, nullable=False, comment="得分")
    correct_count = Column(Integer, nullable=False, comment="答对题数")
    earned_title = Column(String(50), nullable=False, comment="获得称号")
    created_at = Column(DateTime, server_default=func.now(), comment="答题时间")

    user = relationship("User", back_populates="challenge_histories")


# --- 5. 错题本表 ---
class WrongBook(Base):
    __tablename__ = "wrong_book"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    item_name = Column(String(100), nullable=False, comment="做错的题目")
    user_answer = Column(String(50), nullable=False, comment="用户的错误选项")
    correct_answer = Column(String(50), nullable=False, comment="正确答案")
    created_at = Column(DateTime, server_default=func.now(), comment="做错时间")

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

