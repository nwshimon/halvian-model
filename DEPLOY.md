# Put the demo online for your team (Streamlit Community Cloud, free)

Your teammates get a web link and a shared password. Your Groq key lives in a hidden settings box on the hosting site, so nobody else enters or sees a key.

All data is synthetic, so hosting it is safe. Never put real patient information in it.

## Step 1: Build the small copy (2 minutes)

In your project folder (the one with `app.py`), after replacing `app.py` with the newest version:

```
python make_deploy_folder.py
```

This creates a `deploy/` folder of under 1 MB: the code, the trained models, the notes, the fairness report, your saved AI answers (`llm_cache.json`), and only the demo rows of the big history files. It never copies your `.env`. Check it prints `wrote ./deploy/`.

## Step 2: Put the `deploy` folder on GitHub (5 minutes)

1. Create a free account at github.com if you do not have one.
2. Click **New repository**. Name it `halvian-demo`. Choose **Private**. Create it.
3. Easiest way: on the new repo page click **uploading an existing file**, then drag everything INSIDE the `deploy` folder (not the folder itself) into the page, including the `data` and `models` folders. Click **Commit changes**.
   - Or, in Terminal: `cd deploy`, then `git init`, `git add .`, `git commit -m "demo"`, `git branch -M main`, `git remote add origin https://github.com/YOUR_NAME/halvian-demo.git`, `git push -u origin main`.
4. Check the repo does NOT contain a file called `.env`. It should not, but look.

## Step 3: Host it (5 minutes)

1. Go to share.streamlit.io and sign in with GitHub. Allow access to your private repo when asked (it needs this to read the code).
2. Click **New app**. Pick the repo `halvian-demo`, branch `main`, main file `app.py`.
3. Click **Advanced settings**.
   - Python version: choose 3.13 if offered, otherwise the newest.
   - In the **Secrets** box paste this, with your real key and your own password:

```
GROQ_API_KEY = "gsk_your_key_here"
LLM_PROVIDER = "groq"
APP_PASSWORD = "pick-a-team-password"
```

4. Click **Deploy**. The first build takes a few minutes. When it opens you will see a password screen: that means it works.

## Step 4: Share it

Send teammates the web address and the password in a private message, not in a public channel. To change the key or password later: open the app on the hosting site, then **Settings → Secrets**.

## What to expect

| Thing | What happens |
|---|---|
| Sleeping | Free apps sleep after a while unused. The first visitor waits about 30 seconds while it wakes |
| Shared Groq limit | Everyone shares your free Groq limits (about 8 notes a minute, about 1,000 requests a day). If several teammates type new cases at once, some get a rate-limit message and should wait a minute |
| Saved answers | The example cases and the 48 test notes replay from the saved answers with no limit. Only brand-new typed cases call Groq |
| Audit trail | Shared by everyone using the app, and it clears when the app restarts. Fine for a demo |
| Updating | Change code, run `python make_deploy_folder.py` again, upload the new files. The app updates itself |

## Protect your key and your quota

- The password screen stops strangers who find the link from using your key. Keep `APP_PASSWORD` set.
- To stop anyone burning your Groq quota (for example right before recording), add this line to Secrets: `LLM_CACHE_ONLY = "1"`. The app then only replays saved answers; typed cases that were never run before will show an error.
- When the competition is over, delete the key on the Groq console and delete the app.

## If it fails

| Symptom | Fix |
|---|---|
| Build error about a package version | Pick a different Python version in Advanced settings, or send me the error |
| `No such file` for data or models | The `data` or `models` folder was not uploaded: upload them inside the repo |
| Password screen never appears | `APP_PASSWORD` is missing from Secrets (the app is open to anyone with the link) |
| "no API key" in the sidebar | Check the name is exactly `GROQ_API_KEY` in Secrets and save; the app restarts |
| Case demo says rate limit | Wait a minute, or use the saved example cases |