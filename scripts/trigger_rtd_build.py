import os
import requests
import sys

def trigger_build(project_slug, token):
    """
    Triggers a build on ReadTheDocs for the specified project.
    """
    url = f"https://readthedocs.org/api/v3/projects/{project_slug}/builds/"
    headers = {
        "Authorization": f"Token {token}",
        "Content-Type": "application/json",
    }
    data = {
        "branch": "stable", 
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 202:
        print(f"Build triggered successfully for {project_slug}.")
        print(f"Build details: {response.json()}")
    else:
        print(f"Failed to trigger build. Status code: {response.status_code}")
        print(f"Response: {response.text}")
        sys.exit(1)

if __name__ == "__main__":
    # Replace 'helios' with your actual ReadTheDocs project slug if different
    PROJECT_SLUG = "helios" 
    
    # Ensure RTD_TOKEN is set in your environment variables
    TOKEN = os.environ.get("RTD_TOKEN")
    
    if not TOKEN:
        print("Error: RTD_TOKEN environment variable not set.")
        sys.exit(1)
        
    trigger_build(PROJECT_SLUG, TOKEN)
