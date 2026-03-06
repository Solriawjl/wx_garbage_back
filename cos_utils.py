# 配置腾讯云cos
from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client
import os

# ==========================================
# ⚠️ 腾讯云 COS 核心配置 (请替换为你自己的真实数据)
# ==========================================
secret_id = os.environ.get('TENCENT_SECRET_ID')
secret_key = os.environ.get('TENCENT_SECRET_KEY')
REGION = 'ap-chengdu'  # 替换为存储桶地域
BUCKET = 'images-1408449839'  # 替换存储桶完整名称

# 初始化 COS 客户端
config = CosConfig(Region=REGION, SecretId=SECRET_ID, SecretKey=SECRET_KEY)
client = CosS3Client(config)

def upload_file_to_cos(file_bytes: bytes, file_name: str) -> str:
    """
    将文件字节流上传到腾讯云 COS，并返回公网访问 URL
    """
    try:
        # 1. 调用腾讯云接口上传
        response = client.put_object(
            Bucket=BUCKET,
            Body=file_bytes,
            Key=file_name,
            EnableMD5=False
        )

        # 2. 拼接文件的公网访问链接
        # 格式：https://<Bucket>.cos.<Region>.myqcloud.com/<Key>
        file_url = f"https://{BUCKET}.cos.{REGION}.myqcloud.com/{file_name}"
        return file_url

    except Exception as e:
        print(f"COS 上传失败: {e}")
        return None