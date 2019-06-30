from flask import Flask, request, abort

from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import *

app = Flask(__name__)

import os

from ai import AI
from file import File

# Channel Access Token
line_bot_api = LineBotApi('Di/27BUeMNsc9h0MHevLE/qAwqoJz7R+FN9mjasLcm7qbocFR9KddHNyJJwL1maZQdeLCn/bO7+je4ZIA+OU+AjkhtJKdaJ+pFYQXwpksqhjMsdplwX8woI4gapKvej3klD/HaI6jsYEzX2rsC+bHZ36QdB04t89/1O/w1cDnyilFU=+t9Noe+lHmM5v/H+H9PgIP49aty8I1DzH71iAJ3I6QVwA95L5cdW/LXJFD/rT7VKywUpbrC5jKL6mZztUM9ieKtFS5O6hiCldBpThU8cOfPUrqGC3mgzOhzFvz4ds5/He/AdB04t89/1O/w1cDnyilFU=')
# Channel Secret
handler = WebhookHandler('9144184a0089f3ea45a456c7586401e3')

# 監聽所有來自 /callback 的 Post Request
@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']
    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'


# 處理文字訊息
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    msg = event.message.text  # msg 是使用者發過來的String

    message = TextSendMessage(text='哈囉你好，請傳給我一個數字的照片（白底黑字），讓我辨識看看是什麼數字！')
    line_bot_api.reply_message(event.reply_token, message)
    print('-----------------')

# 處理影音訊息
@handler.add(MessageEvent, message=(ImageMessage, VideoMessage, AudioMessage))
def handle_content_message(event):
    is_image = False
    if isinstance(event.message, ImageMessage):
        ext = 'jpg'
        is_image = True
    elif isinstance(event.message, VideoMessage):
        ext = 'mp4'
    elif isinstance(event.message, AudioMessage):
        ext = 'm4a'
    else:
        return

    if is_image == False:
        line_bot_api.reply_message(event.reply_token, '這好像不是圖片唷')
    else:
        message_content = line_bot_api.get_message_content(event.message.id)
        img, file_path = file.save_bytes_image(message_content.content)
        pred = ai.predict_image_with_path(file_path)

        line_bot_api.reply_message(
            event.reply_token, [
                TextSendMessage(text=pred)
            ])

ai = AI()
file = File()

if not os.path.exists('model/'):
    os.makedirs('model/')
if not os.path.exists('media/'):
    os.makedirs('media/')

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
