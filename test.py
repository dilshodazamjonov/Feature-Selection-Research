import os

import dotenv

dotenv.load_dotenv()

if __name__ == "__main__":
    print("OPENAI_API_KEY configured:", bool(os.getenv("OPENAI_API_KEY")))
