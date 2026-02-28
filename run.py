"""快速启动脚本 - 禁用不必要的警告以加快启动速度"""
import os

# 禁用 TensorFlow 警告和 oneDNN 优化提示
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# 禁用 tokenizers 并行警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    import uvicorn
    from config.settings import settings

    print("=" * 50)
    print("桌游规则问答引擎启动中...")
    print(f"访问地址: http://localhost:{settings.API_PORT}")
    print(f"兼容地址: http://127.0.0.1:{settings.API_PORT}")
    print("=" * 50)

    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=False  # 关闭热重载加快启动
    )
