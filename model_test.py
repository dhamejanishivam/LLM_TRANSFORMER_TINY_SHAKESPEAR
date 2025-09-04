HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VISION</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        html, body {
            height: 100%;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background-color: #212121;
            color: #ffffff;
            overflow: hidden; /* Control overflow from the main container */
            display: flex;
            flex-direction: column;
            font-size: 17px; /* <<< Adjusted Font Size Here (from 16px to 17px) */
        }

        .app-container {
            display: flex;
            flex-direction: column;
            height: 100vh;
            background-color: #212121;
            overflow: hidden; /* Prevents unwanted scrollbars */
            position: relative; /* For welcome section positioning */
        }

        /* Main Header - Simplified */
        .main-header {
            padding: 12px 16px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            display: flex;
            align-items: center;
            justify-content: center;
            background-color: #212121;
            min-height: 50px;
            color: #ececec;
            position: relative; /* Z-index for header */
            z-index: 100;
        }

        .model-selector {
            font-weight: 500;
            font-size: 18px;
        }

        /* Initial Centering for Welcome Screen */
        .welcome-screen-wrapper {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            transition: opacity 0.5s ease; /* Smooth fade out */
            z-index: 50; /* Ensure it's above message container initially */
            background-color: #212121; /* Match body background */
        }

        .welcome-screen-wrapper.hidden {
            opacity: 0;
            pointer-events: none; /* Disable interaction when hidden */
        }

        .welcome-title {
            font-size: 32px;
            font-weight: 400;
            margin-bottom: 32px;
            color: #ececec;
            text-align: center;
        }

        /* Messages Area */
        .messages-container {
            flex: 1;
            overflow-y: auto;
            position: absolute; /* Take full space, then content scrolls */
            top: 50px; /* Below header */
            left: 0;
            right: 0;
            bottom: 70px; /* Above input section */
            padding: 20px 0;
            display: flex;
            flex-direction: column;
            opacity: 0; /* Hidden initially */
            transition: opacity 0.5s ease; /* Smooth fade in */
            z-index: 1; /* Below welcome screen initially */
        }

        .messages-container.visible {
            opacity: 1;
        }

        .messages-wrapper {
            max-width: 900px; /* <<< Increased Width Here */
            margin: 0 auto;
            padding: 0 32px;
            width: 100%;
            transition: opacity 0.3s ease;
        }

        .message-group {
            margin-bottom: 20px;
            display: flex;
            flex-direction: column;
            width: 100%;
        }

        .message {
            display: flex;
            line-height: 1.6;
            font-size: 1rem; /* Use rem for consistency with body font size */
            width: fit-content;
            max-width: 100%;
        }

        .message.user {
            background-color: #323232d9;
            color: #ffffff;
            padding: 12px 16px;
            border-radius: 18px;
            align-self: flex-end;
            margin-right: 0;
        }

        .message.assistant {
            background-color: transparent;
            color: #ffffff;
            padding: 0;
            align-self: flex-start;
            margin-left: 0;
             font-size: 1.15rem !important; 
        }

        .message-content {
            flex: 1;
            white-space: pre-wrap;
            padding-top: 2px;
        }

        /* Thinking Animation */
        .thinking-container {
            display: flex;
            align-items: center;
            gap: 4px;
            color: #8e8ea0;
            font-size: 1rem; /* Use rem for consistency */
            width: fit-content;
            padding: 12px 0;
        }

        .thinking-dot {
            width: 6px;
            height: 6px;
            background-color: #8e8ea0;
            border-radius: 50%;
            opacity: 0.2;
            animation: chatgpt-pulse 1.5s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        }

        .thinking-dot:nth-child(1) { animation-delay: 0s; }
        .thinking-dot:nth-child(2) { animation-delay: 0.1s; }
        .thinking-dot:nth-child(3) { animation-delay: 0.2s; }

        @keyframes chatgpt-pulse {
            0%, 100% { opacity: 0.2; }
            50% { opacity: 1; }
        }

        /* Input Area */
        .input-section {
            padding: 16px 24px 24px;
            background-color: #212121;
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            z-index: 10;
            transition: transform 0.5s ease-out;
            display: flex;
            justify-content: center;
        }

        .input-section::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 30px;
            background: linear-gradient(to top, rgba(33, 33, 33, 1), rgba(33, 33, 33, 0));
            pointer-events: none;
            transform: translateY(-100%);
        }

        .input-container {
            max-width: 900px; /* <<< Increased Width Here */
            margin: 0 auto;
            position: relative;
            width: 100%;
        }

        .input-wrapper {
            background-color: #303030;
            border: none;
            border-radius: 28px;
            display: flex;
            align-items: flex-end;
            padding: 12px;
            position: relative;
            transition: background-color 0.2s ease;
        }

        .input-wrapper:focus-within {
            background-color: #3a3a3a;
        }

        .message-input {
            flex: 1;
            background: transparent;
            border: none;
            outline: none;
            color: #ffffff;
            font-size: 1rem; /* Use rem for consistency */
            font-family: inherit;
            resize: none;
            max-height: 200px;
            min-height: 24px;
            line-height: 24px;
            overflow-y: auto;
            padding-right: 48px;
            scrollbar-width: thin;
            scrollbar-color: #4d4d4f transparent;
        }

        .message-input::-webkit-scrollbar {
            width: 8px;
        }

        .message-input::-webkit-scrollbar-track {
            background: transparent;
        }

        .message-input::-webkit-scrollbar-thumb {
            background: #4d4d4f;
            border-radius: 4px;
        }

        .message-input::-webkit-scrollbar-thumb:hover {
            background: #676775;
        }

        /* Send Button */
        .send-btn {
            width: 36px;
            height: 36px;
            border: none;
            border-radius: 50%;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, #4d4d4f 0%, #676775 100%);
            background: #fff;
            transition: background 0.2s ease, transform 0.2s ease;
            position: absolute;
            right: 8px;
            bottom: 8px;
        }

        .send-btn:hover {
            background: linear-gradient(135deg, #676775 0%, #fff 100%);
            transform: scale(1.05);
        }

        .send-btn:active {
            transform: scale(0.95);
        }

        .send-btn:disabled {
            background: #4d4d4f;
            transform: none;
            opacity: 0.6;
        }

        .send-btn svg {
            width: 18px;
            height: 18px;
            fill: none;
            stroke: white;
            stroke-width: 2.5;
            stroke-linecap: round;
            stroke-linejoin: round;
            transform: translate(1px, -1px) rotate(45deg);
        }

        .send-btn:disabled svg {
            stroke: #b0b0b0;
        }

        /* Mobile Responsive */
        @media (max-width: 768px) {
            .messages-wrapper {
                padding: 0 16px;
            }

            .input-section {
                padding: 13px;
            }
        }

        /* Scrollbar Styling for messages-container */
        ::-webkit-scrollbar {
            width: 8px;
        }

        ::-webkit-scrollbar-track {
            background: transparent;
        }

        ::-webkit-scrollbar-thumb {
            background: #4d4d4f;
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: #676775;
        }

        /* Firefox scrollbar */
        * {
            scrollbar-width: thin;
            scrollbar-color: #4d4d4f transparent;
        }
    </style>
</head>
<body>
    <div class="app-container">
        <div class="main-header">
            <div class="model-selector">
                VISION
            </div>
        </div>

        <div class="welcome-screen-wrapper" id="welcomeScreenWrapper">
            <h1 class="welcome-title">How can I help you today?</h1>
            <div class="input-container" style="position: static;">
                <div class="input-wrapper" id="centeredInputWrapper">
                    <textarea 
                        class="message-input" 
                        id="messageInputInitial" 
                        placeholder="Aurther said, wrecking the laws of his nature's reason..."
                        rows="1"
                        onkeydown="handleKeyPress(event, true)"
                        oninput="adjustTextareaHeight(true)"
                    >Aurther said, wrecking the laws of his nature's reason...</textarea>
                    <button class="send-btn" id="sendBtnInitial" onclick="sendMessage(true)">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                            <path d="M7 11L12 6L17 11M12 18V7" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"></path>
                        </svg>
                    </button>
                </div>
            </div>
        </div>

        <div class="messages-container" id="messagesContainer">
            <div class="messages-wrapper" id="messagesWrapper">
                <div id="messagesArea"></div>
            </div>
        </div>

        <div class="input-section" id="inputSectionBottom">
            <div class="input-container">
                <div class="input-wrapper" id="inputWrapperBottom">
                    <textarea 
                        class="message-input" 
                        id="messageInputBottom" 
                        placeholder="Message VISION..."
                        rows="1"
                        onkeydown="handleKeyPress(event, false)"
                        oninput="adjustTextareaHeight(false)"
                    ></textarea>
                    
                    <button class="send-btn" id="sendBtnBottom" onclick="sendMessage(false)">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                            <path d="M7 11L12 6L17 11M12 18V7" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"></path>
                        </svg>
                    </button>
                </div>
            </div>
        </div>
    </div>

    <script>
        const welcomeScreenWrapper = document.getElementById('welcomeScreenWrapper');
        const messagesContainer = document.getElementById('messagesContainer');
        const messagesArea = document.getElementById('messagesArea');
        const inputSectionBottom = document.getElementById('inputSectionBottom');
        
        const messageInputInitial = document.getElementById('messageInputInitial');
        const sendBtnInitial = document.getElementById('sendBtnInitial');

        const messageInputBottom = document.getElementById('messageInputBottom');
        const sendBtnBottom = document.getElementById('sendBtnBottom');

        let isGenerating = false;
        let isFirstMessage = true;

        // Determine active input and button based on first message state
        function getActiveInputElements() {
            if (isFirstMessage) {
                return { input: messageInputInitial, sendBtn: sendBtnInitial };
            } else {
                return { input: messageInputBottom, sendBtn: sendBtnBottom };
            }
        }
        
        function adjustTextareaHeight(isInitial) {
            const { input } = getActiveInputElements();
            input.style.height = 'auto';
            input.style.height = Math.min(input.scrollHeight, 200) + 'px';
            updateSendButton();
        }

        function updateSendButton() {
            const { input, sendBtn } = getActiveInputElements();
            const hasText = input.value.trim().length > 0;
            sendBtn.disabled = !hasText || isGenerating;
        }

        async function transitionToChatMode() {
            if (isFirstMessage) {
                isFirstMessage = false;
                
                // Move content from initial input to bottom input
                messageInputBottom.value = messageInputInitial.value;
                messageInputInitial.value = '';
                adjustTextareaHeight(false); // Adjust bottom textarea height

                welcomeScreenWrapper.classList.add('hidden');
                messagesContainer.classList.add('visible'); // Fade in message container
                
                // Focus the bottom input field immediately after transition starts
                messageInputBottom.focus();
            }
        }

        function addMessage(content, isUser = false) {
            const messageGroup = document.createElement('div');
            messageGroup.className = 'message-group';

            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user' : 'assistant'}`;
            
            const messageContent = document.createElement('div');
            messageContent.className = 'message-content';
            messageContent.innerHTML = content;
            
            messageDiv.appendChild(messageContent);
            messageGroup.appendChild(messageDiv);
            
            messagesArea.appendChild(messageGroup);
            scrollToBottom();
            return messageContent;
        }

        function addThinkingIndicator() {
            const messageGroup = document.createElement('div');
            messageGroup.className = 'message-group';
            messageGroup.id = 'thinkingGroup';
            
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message assistant';
            
            const messageContent = document.createElement('div');
            messageContent.className = 'message-content';
            messageContent.innerHTML = `
                <div class="thinking-container">
                    <div class="thinking-dot"></div>
                    <div class="thinking-dot"></div>
                    <div class="thinking-dot"></div>
                </div>
            `;
            
            messageDiv.appendChild(messageContent);
            messageGroup.appendChild(messageDiv);
            
            messagesArea.appendChild(messageGroup);
            scrollToBottom();
        }

        function removeThinkingIndicator() {
            const thinkingGroup = document.getElementById('thinkingGroup');
            if (thinkingGroup) {
                thinkingGroup.remove();
            }
        }

        function scrollToBottom() {
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        async function typeMessage(element, text) {
            const chars = text.split('');
            element.textContent = '';
            return new Promise(resolve => {
                let i = 0;
                const interval = setInterval(() => {
                    if (i < chars.length) {
                        element.textContent += chars[i];
                        scrollToBottom();
                        i++;
                    } else {
                        clearInterval(interval);
                        resolve();
                    }
                }, 5); // Typing speed variable (milliseconds per character)
            });
        }

        async function sendMessage(isInitialCall) {
            const { input } = getActiveInputElements(); // Get the active input field
            const message = input.value.trim();
            input.value = ''; // Clear the active input field's value
            input.style.height = 'auto'; // Reset height
            
            if (!message || isGenerating) {
                return;
            }

            if (isInitialCall) {
                await transitionToChatMode();
            }

            addMessage(message, true); // Add user message
            
            
            isGenerating = true;
            updateSendButton();
            messageInputInitial.disabled = true; // Disable both for safety
            messageInputBottom.disabled = true;
            
            addThinkingIndicator();

            try {
                await new Promise(resolve => setTimeout(resolve, 500)); 

                const url = `/predict?message=${encodeURIComponent(message)}`;
                const response = await fetch(url);

                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.error || 'Something went wrong on the server.');
                }

                const data = await response.json();
                
                removeThinkingIndicator();
                const assistantMessageContentDiv = addMessage('', false);
                await typeMessage(assistantMessageContentDiv, data.response);
                
            } catch (error) {
                console.error('Error fetching AI response:', error);
                removeThinkingIndicator();
                addMessage(`Error: ${error.message}. Please check if the model is running correctly.`, false);
            } finally {
                isGenerating = false;
                updateSendButton();
                messageInputInitial.disabled = false;
                messageInputBottom.disabled = false;
                getActiveInputElements().input.focus(); // Focus the currently active input
            }
        }

        function handleKeyPress(event, isInitialCall) {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                sendMessage(isInitialCall);
            }
        }

        // Event listeners
        // Listen to both inputs' 'input' event, the getActiveInputElements will determine which one to check
        messageInputInitial.addEventListener('input', () => updateSendButton());
        messageInputBottom.addEventListener('input', () => updateSendButton());
        
        // Initial setup on load
        window.addEventListener('load', () => {
            adjustTextareaHeight(true); // Adjust initial textarea height
            messageInputInitial.focus(); // Focus the initial input
            updateSendButton(); // Set initial button state
        });

        window.addEventListener('resize', () => {
            if (!isFirstMessage) {
                scrollToBottom();
            }
        });
    </script>
</body>
</html>
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


# =============== Hyperparameters =============== 
batch_size = 64 # How many independent sequence will we process in parallel
block_size = 256 # what is the maximum context length for prediction?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
vocab_size = 107






class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self,head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        # Compute attention scores ('affinities')
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,C) @ (B,C,T) ---> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self.dropout(wei)
        # Perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B,T,T) @ (B,T,C) --> (B,T,C)
        return out
    


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads],dim=-1)
        out = self.proj(out)
        return out 
    


class FeedForward(nn.Module):
    """ a simple linear layer followed by a non_lin   earity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self,x):
        return self.net(x)



class Block(nn.Module):
    """ Transformer block : communicates followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd : embedding dimension, n_head: the number of heads we would like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)


    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x



class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.positional_embedding_table = nn.Embedding(block_size, n_embd)

        # self.blocks = nn.Sequential(
        #     Block(n_embd,n_head=4),
        #     Block(n_embd,n_head=4),
        #     Block(n_embd,n_head=4),
        #     nn.LayerNorm(n_embd)
        # )

        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    

    def forward(self, idx, targets=None):
        
        B,T = idx.shape

        token_emb = self.token_embedding_table(idx)
        pos_emb = self.positional_embedding_table(torch.arange(T, device=device))
        x = token_emb + pos_emb # (B,T,C)
        # x = self.sa_head(x)
        # x = self.ffwd(x) # (B,T,C)
        x = self.blocks(x)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
         
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        

        return logits, loss
    

    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            # Getting the predictions
            logits, loss = self(idx_cond)
            # Focus only on the last time step
            logits = logits[:, -1, :]
            # apply softmaxx to get probabilities
            probs = F.softmax(logits, dim=1)
            # samlpe from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # 
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


import torch, pickle
path = 'model_dir'
with open(f"{path}/texter.pkl", "rb") as f:
    meta = pickle.load(f)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Rebuild model with exact same hyperparams
model = BigramLanguageModel().to(device)
model.load_state_dict(torch.load(f"{path}/texter.pth", map_location=device))
model.eval()

# Restore tokenizer
stoi, itos = meta['stoi'], meta['itos']

def encode(s):
    lis =[]
    for c in s:
        lis.append(stoi[c])
    return lis
# Decode helper
def decode(l): return ''.join([itos[i] for i in l])



def chat_with_model(user_input_raw,max_new_tokens=100):
  new = int(len(user_input_raw))*5
  if new>100:
    new = min(new,500)
    max_new_tokens=new
  user_input = encode(user_input_raw)
  user_input = torch.tensor(user_input, dtype=torch.long).unsqueeze(0).to(device)
  output = model.generate(idx=user_input,max_new_tokens=50)
  output = output[0].tolist()
  ans = decode(output)
  ans=ans.replace(user_input_raw,'')
  # print(f'You asked : {user_input_raw} \n AI reponse : {ans}')
  print(f'{ans}')
  return ans



from flask import Flask, request, jsonify
from threading import Thread

# --- Flask App Setup ---
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return HTML_TEMPLATE  # Keep using your existing HTML_TEMPLATE

@app.route('/predict', methods=['GET'])
def predict_route():
    message = request.args.get('message')
    if not message:
        return jsonify({'error': 'No message provided'}), 400

    ai_response = chat_with_model(message)
    return jsonify({'response': ai_response}), 200

def run_app():
    app.run(host='0.0.0.0', port=4998, debug=True, use_reloader=False)


if __name__ == '__main__':
    Thread(target=run_app).start()