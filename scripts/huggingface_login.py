"""
Login to Hugging Face using the API key stored in the .env file
or infisical secrets store if the user is not logged in to Hugging Face
"""
import os
import dotenv
from infisical_sdk import InfisicalSDKClient # type: ignore
from huggingface_hub import login, whoami, scan_cache_dir


def get_secret(secret_name):
    """
    Retrieve a secret from the secrets store or 
    infisical secrets store if the user is not 
    logged in to Hugging Face

    Args:
        secret_name: The name of the secret to retrieve
    Returns:
        The secret value
    """
    if os.path.exists("../.env"):
        dotenv.load_dotenv()
        if secret_name in os.environ:
            return os.getenv(secret_name)
        if os.getenv("SECRET_CLIENT_ID") and os.getenv("SECRET_CLIENT_SECRET"):
            # Initialize the client
            client = InfisicalSDKClient(host="https://app.infisical.com")
            client.auth.universal_auth.login(
                client_id=os.getenv("SECRET_CLIENT_ID"), 
                client_secret=os.getenv("SECRET_CLIENT_SECRET"),
            )
            # Use the SDK to interact with Infisical
            secret = client.secrets.get_secret_by_name(
                secret_name=secret_name, 
                environment_slug=os.getenv("SECRET_ENVIRONMENT_SLUG"),
                secret_path=os.getenv("SECRET_PATH"),
                project_slug=os.getenv("SECRET_PROJECT_SLUG"),
            )
            return secret.secretValue
        else:
            raise ValueError("Secret not found")
    else:
        raise ValueError("No .env file found")

def logged_in():
    """
    Check if the user is logged in to Hugging Face
    Returns:
        True if the user is logged into huggingface, False otherwise
    """
    user = whoami()
    if user:
        return True
    return False

if not logged_in():
    huggingface_token = get_secret("HUGGINGFACE_APIKEY")
    if huggingface_token:
        login(huggingface_token)
        print("Hugging Face logged in successfully.")
    else:
        print("Hugging Face token not found.")
else:
    print("Hugging Face already logged in.")

print("Repos in cache:")
cache = scan_cache_dir()
repo_ids = [repo.repo_id for repo in cache.repos]
for repo_id in repo_ids:
    print(repo_id)