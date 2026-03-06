# 数据库配置
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# 格式：mysql+pymysql://用户名:密码@主机地址:端口/数据库名?charset=utf8mb4
SQLALCHEMY_DATABASE_URL = "mysql+pymysql://root:5285280@127.0.0.1:3306/garbage?charset=utf8mb4"

# 创建数据库引擎
engine = create_engine(SQLALCHEMY_DATABASE_URL)

# 创建数据库会话类
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 创建模型基类
Base = declarative_base()

# 获取数据库会话的依赖函数（后续每个接口都要用到）
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()