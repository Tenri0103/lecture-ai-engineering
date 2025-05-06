# AWSチャットアプリ用カスタムLLM API実装ガイド

このガイドでは、AWS環境上でBedrockの代わりに自作のLLM APIを使用するチャットアプリケーションの実装方法について説明します。

## 目次

1. [概要](#概要)
2. [環境構築](#環境構築)
3. [Google Colab上でのAPI実装](#google-colab上でのapi実装)
4. [AWS Lambdaの修正](#aws-lambdaの修正)
5. [デプロイと動作確認](#デプロイと動作確認)
6. [GitHubへの提出](#githubへの提出)
7. [トラブルシューティング](#トラブルシューティング)

## 概要

この演習では、AWS CDKを使用してデプロイされたチャットアプリケーションを修正し、Amazon Bedrockの代わりに自作のLLM APIを使用するように変更します。主な実装ステップは以下の通りです：

1. Google Colab上でFast APIを使用して推論用APIを実装
2. AWS Lambdaのコードを修正して自作APIを呼び出すように変更
3. 修正したコードをデプロイして動作確認
4. 変更をGitHubリポジトリにコミット

## 環境構築

### 前提条件

- AWSアカウントへのアクセス権限
- 第1回演習で使用したGoogle Colabの環境
- GitHubアカウント（リポジトリをフォークするため）

### リポジトリのフォーク

1. [https://github.com/keisskaws/simplechat](https://github.com/keisskaws/simplechat) にアクセス
2. 右上の「Fork」ボタンをクリックしてリポジトリをフォーク
3. フォークしたリポジトリをローカルにクローン（または後述のCloudShellで直接クローン）

```bash
git clone https://github.com/[あなたのGitHubユーザー名]/simplechat.git
```

## Google Colab上でのAPI実装

### 基本的なFast APIの実装

以下のコードをGoogle Colabに実装します。このコードは、テキスト生成モデルを使用して簡単な推論APIを提供します。

```python
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import nest_asyncio
from pyngrok import ngrok

# Fast APIのセットアップ
app = FastAPI()

# CORSの設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
async def chat(request: Request):
    # リクエストボディの取得
    data = await request.json()
    message = data.get("message", "")
    conversation_history = data.get("conversationHistory", [])
    
    print(f"受信メッセージ: {message}")
    print(f"会話履歴: {conversation_history}")
    
    # ここで実際のモデル推論を行います
    # この例では簡単のためにエコーレスポンスを返していますが、
    # 実際には自分の選んだモデルで推論を行う処理を実装します
    response = f"あなたのメッセージ「{message}」を受け取りました。これはカスタムAPIからの応答です。"
    
    return {"response": response}

# ngrokでトンネリング
ngrok_tunnel = ngrok.connect(8000)
print(f"Public URL: {ngrok_tunnel.public_url}")

# APIサーバー起動
nest_asyncio.apply()
uvicorn.run(app, port=8000)
```

### モデルを使用した実装例

実際に言語モデルを使用する場合は、以下のように実装できます（例として transformers を使用）：

```python
# 必要なライブラリをインストール
!pip install transformers torch fastapi uvicorn pyngrok nest_asyncio

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import nest_asyncio
from pyngrok import ngrok
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# モデルのロード（選択したモデルに応じて変更してください）
model_name = "rinna/japanese-gpt-neox-3.6b"  # 例として日本語モデル
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Fast APIのセットアップ
app = FastAPI()

# CORSの設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    message = data.get("message", "")
    
    # プロンプトの作成
    prompt = f"ユーザー: {message}\nAI: "
    
    # モデルによる推論
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=200,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 出力テキストのデコード
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # "AI: "以降の部分を抽出
    response = generated_text.split("AI: ")[-1].strip()
    
    return {"response": response}

# ngrokでトンネリング
ngrok_tunnel = ngrok.connect(8000)
print(f"Public URL: {ngrok_tunnel.public_url}")
print("このURLをLambda関数に設定してください")

# APIサーバー起動
nest_asyncio.apply()
uvicorn.run(app, port=8000)
```

**重要**: ngrokで生成されるPublic URLをメモしておきます。このURLはLambda関数の修正時に使用します。

## AWS Lambdaの修正

AWS Lambdaコード（lambda/index.py）を修正して、Amazon Bedrockの代わりに自作APIを呼び出すようにします。

### 修正前のコード確認

まず、元のコードでBedrockを呼び出している部分を特定します。通常、boto3クライアントを使用してBedrockを呼び出している部分があります。

### lambda/index.py の修正例

```python
import json
import os
import urllib.request
import urllib.error
import urllib.parse

def handler(event, context):
    print(f"Event: {event}")
    
    try:
        # リクエストの解析
        body = json.loads(event.get('body', '{}'))
        message = body.get('message', '')
        conversation_history = body.get('conversationHistory', [])
        
        # カスタムAPIのエンドポイント（Google Colabで表示されたURLに/chatを追加）
        api_endpoint = "https://xxxx-xx-xxx-xxx-xx.ngrok.io/chat"  # ここを自分のngrok URLに変更
        
        # リクエストデータの準備
        data = {
            "message": message,
            "conversationHistory": conversation_history
        }
        
        # JSONデータをエンコード
        request_data = json.dumps(data).encode('utf-8')
        
        # HTTPリクエストの作成
        req = urllib.request.Request(
            api_endpoint,
            data=request_data,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        
        # APIリクエストの送信
        try:
            with urllib.request.urlopen(req, timeout=25) as response:
                response_data = response.read()
                response_json = json.loads(response_data)
                
                # 応答の取得
                ai_response = response_json.get('response', '')
                
            # 成功レスポンスの返却
            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
                    'Access-Control-Allow-Methods': 'OPTIONS,POST'
                },
                'body': json.dumps({
                    'success': True,
                    'response': ai_response
                })
            }
        except urllib.error.URLError as e:
            print(f"APIリクエストエラー: {str(e)}")
            return {
                'statusCode': 500,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'success': False,
                    'message': f"カスタムAPIへの接続に失敗しました: {str(e)}"
                })
            }
            
    except Exception as e:
        print(f"エラー: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'success': False,
                'message': str(e)
            })
        }
```

**重要**: `api_endpoint`変数にはGoogle Colabで表示されたngrokのURLを設定してください。URLの末尾に`/chat`を追加するのを忘れないでください。

## デプロイと動作確認

### CDKを使用したデプロイ

AWS CloudShellまたは自分のPC環境からデプロイを行います。

```bash
# リポジトリのディレクトリに移動
cd simplechat

# lambda/index.pyを修正した後
cdk deploy
```

デプロイが完了したら、出力されるCloudFrontのURLにアクセスしてアプリケーションの動作を確認します。

### 動作確認のポイント

1. アプリケーションにログインできるか確認
2. テキストを入力して送信し、カスタムAPIからの応答が表示されるか確認
3. エラーが発生した場合はCloudWatchのログを確認

## GitHubへの提出

動作確認ができたら、変更をGitHubリポジトリにコミットします。

```bash
# 変更をステージング
git add lambda/index.py

# コミット
git commit -m "Replace Bedrock with custom API"

# GitHubにプッシュ
git push origin main  # または master（デフォルトブランチ名に合わせてください）
```

### 提出チェックリスト

- [x] Google Colab上でFast APIを使用したAPIの実装
- [x] lambda/index.pyの修正
- [x] デプロイと動作確認
- [x] 変更のGitHubへのコミット
- [x] リポジトリURLの提出準備

## トラブルシューティング

### Google ColabのAPIに接続できない場合

1. ngrokのURLが正しく設定されているか確認
2. Colabのランタイムがアクティブであることを確認
3. ngrokの無料プランは接続時間に制限があるため、長時間経過した場合は再起動が必要

### デプロイエラーの場合

1. CDKのエラーメッセージを確認
2. lambda/index.pyの構文エラーがないか確認
3. 必要に応じてCloudShellのログを確認

### アプリケーションが正しく応答しない場合

1. CloudWatchでLambda関数のログを確認
2. lambda/index.pyのapi_endpointが正しく設定されているか確認
3. Google ColabのAPIがリクエストを正しく処理できているか確認

---

この演習では、AWS環境のサーバーレスアプリケーションを修正して、Amazon Bedrockの代わりに自作のLLM APIを使用するように変更しました。この経験を通じて、AWSのサーバーレスアーキテクチャとカスタムAPIの統合方法についての理解を深めることができます。
