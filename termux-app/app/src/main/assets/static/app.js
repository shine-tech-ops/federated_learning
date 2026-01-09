class SimpleOllamaChat {
    constructor() {
        // 创建 axios 实例，连接到本地 Ollama
        this.client = axios.create({
            baseURL: `http://${window.ollamaHost}`,
            timeout: 300 * 1000 // 超时时间
        });

        // 存储对话历史
        this.messages = [];
    }

    instance() {
        this.client = axios.create({
            baseURL: `http://${window.ollamaHost}`,
            timeout: 300 * 1000 // 超时时间
        });
    }

    /**
     * 发送消息给 Ollama
     * @param {string} userInput - 用户输入
     * @param {string} model - 使用的模型，默认 llama2
     * @returns {Promise<string>} - 助手回复
     */
    async sendMessage(userInput, model = 'llama2') {
        try {
            // 1. 添加用户消息到历史
            this.messages.push({
                role: 'user',
                content: userInput
            });

            // 2. 准备发送的数据
            const requestData = {
                model: model,
                messages: this.messages,
                stream: false // 简单起见，不使用流式
            };

            // 3. 发送请求到 Ollama
            this.instance();
            console.log(this.client);
            const response = await this.client.post('/api/chat', requestData);

            // 4. 获取助手回复
            const assistantReply = response.data.message.content;

            // 5. 添加助手回复到历史
            this.messages.push({
                role: 'assistant',
                content: assistantReply
            });

            // 6. 返回回复
            return assistantReply;

        } catch (error) {
            // 错误处理
            if (error.code === 'ECONNREFUSED') {
                throw new Error('无法连接到 Ollama，请确保 Ollama 服务正在运行\n运行命令: ollama serve');
            }

            if (error.response) {
                throw new Error(`Ollama 返回错误: ${error.response.status} - ${error.response.data.error}`);
            }

            throw new Error(`请求失败: ${error.message}`);
        }
    }

    /**
     * 清空对话历史
     */
    clearHistory() {
        this.messages = [];
        console.log('对话历史已清空');
    }

    /**
     * 查看当前对话历史
     */
    showHistory() {
        console.log('\n=== 对话历史 ===');
        this.messages.forEach((msg, index) => {
            const role = msg.role === 'user' ? '你' : '助手';
            console.log(`${index + 1}. ${role}: ${msg.content}`);
        });
        console.log('================\n');
    }

    /**
     * 设置系统提示（可选）
     * @param {string} prompt - 系统提示词
     */
    setSystemPrompt(prompt) {
        // 添加系统消息到对话开头
        this.messages.unshift({
            role: 'system',
            content: prompt
        });
    }
}

class CouldChat {
    constructor(apiKey) {
        this.apiKey = apiKey || 'sk-sjvqTxWynT1Y5isDZ0Rs78eIGLsSCSLYlWbEKxNGXOuAuWsF';
        this.baseURL = 'https://api.chatanywhere.tech/v1';

        // 创建 axios 实例
        this.client = axios.create({
            baseURL: this.baseURL,
            headers: {
                'Authorization': `Bearer ${this.apiKey}`,
                'Content-Type': 'application/json'
            },
            timeout: 300 * 1000 // 超时时间
        });

        // 存储对话历史
        this.messages = [];
    }

    /**
     * 发送消息到 ChatAnywhere API
     * @param {string} userInput - 用户输入
     * @param {string} model - 使用的模型，默认 gpt-3.5-turbo
     * @param {number} maxTokens - 最大 token 数
     * @param {number} temperature - 温度参数
     * @returns {Promise<string>} - 助手回复
     */
    async sendMessage(userInput, model = 'gpt-3.5-turbo', maxTokens = 2000, temperature = 0.7) {
        try {
            // 1. 添加用户消息到历史
            this.messages.push({
                role: 'user',
                content: userInput
            });

            // 2. 准备发送的数据
            const requestData = {
                model: model,
                messages: this.messages,
                max_tokens: maxTokens,
                temperature: temperature,
                stream: false
            };

            // 3. 发送请求到 ChatAnywhere API
            const response = await this.client.post('/chat/completions', requestData);

            // 4. 获取助手回复
            const assistantReply = response.data.choices[0].message.content;

            // 5. 添加助手回复到历史
            this.messages.push({
                role: 'assistant',
                content: assistantReply
            });

            // 6. 返回回复
            return assistantReply;

        } catch (error) {
            // 错误处理
            if (error.response) {
                const status = error.response.status;
                const data = error.response.data;

                switch (status) {
                    case 401:
                        throw new Error('API 密钥无效或已过期，请检查您的 API 密钥');
                    case 429:
                        throw new Error('请求过于频繁，请稍后再试');
                    case 400:
                        throw new Error(`请求参数错误: ${data.error?.message || '未知错误'}`);
                    case 404:
                        throw new Error('请求的模型不存在或不可用');
                    default:
                        throw new Error(`ChatAnywhere API 返回错误: ${status} - ${data.error?.message || '未知错误'}`);
                }
            }

            if (error.code === 'ECONNREFUSED' || error.code === 'ENETUNREACH') {
                throw new Error('无法连接到 ChatAnywhere API，请检查网络连接');
            }

            throw new Error(`请求失败: ${error.message}`);
        }
    }

    /**
     * 清空对话历史
     */
    clearHistory() {
        this.messages = [];
        const chatMessages = document.getElementById('chatMessages');
        const flexSeries = chatMessages.querySelectorAll('.flex')
        flexSeries.forEach((item, index) => {
            if (index !== 0) {
                item.remove();
            }
        });
        console.log('对话历史已清空');
    }

    /**
     * 查看当前对话历史
     */
    showHistory() {
        console.log('\n=== ChatAnywhere 对话历史 ===');
        this.messages.forEach((msg, index) => {
            const role = msg.role === 'user' ? '你' : '助手';
            console.log(`${index + 1}. ${role}: ${msg.content}`);
        });
        console.log('===========================\n');
    }

    /**
     * 设置系统提示
     * @param {string} prompt - 系统提示词
     */
    setSystemPrompt(prompt) {
        // 查找是否已有系统消息
        const systemMessageIndex = this.messages.findIndex(msg => msg.role === 'system');

        if (systemMessageIndex >= 0) {
            // 更新已有的系统消息
            this.messages[systemMessageIndex].content = prompt;
        } else {
            // 添加系统消息到对话开头
            this.messages.unshift({
                role: 'system',
                content: prompt
            });
        }
    }

    /**
     * 获取可用的模型列表
     * @returns {Promise<Array>} - 模型列表
     */
    async getAvailableModels() {
        try {
            const response = await this.client.get('/models');
            return response.data.data;
        } catch (error) {
            console.error('获取模型列表失败:', error);
            throw error;
        }
    }

    /**
     * 设置 API 密钥
     * @param {string} apiKey - 新的 API 密钥
     */
    setApiKey(apiKey) {
        this.apiKey = apiKey;
        this.client.defaults.headers['Authorization'] = `Bearer ${apiKey}`;
    }

    /**
     * 获取当前对话历史
     * @returns {Array} - 对话历史数组
     */
    getHistory() {
        return [...this.messages];
    }

    /**
     * 设置对话历史
     * @param {Array} messages - 新的对话历史
     */
    setHistory(messages) {
        this.messages = [...messages];
    }
}

class FederatedLearningApp {
    constructor() {
        this.currentModel = window.platform === 'ios' ? 'large' : 'small';
        this.currentLargeModel = 'gpt-4';
        this.currentPlatform = 'could';
        this.isTraining = false;
        this.trainingProgress = 0;
        this.trainingCount = 0;
        this.dialogCount = 0;
        this.uploadCount = 0;
        this.chatHistory = [];
        this.currentTaskId = '1';
        this.samplePath = '';
        this.customMessageTimeout = undefined;
        this.localChat = new SimpleOllamaChat();
        this.couldChat = new CouldChat();
        // this.init();
    }

    init() {
        this.bindTaskEvents();
        this.bindLargeModelEvents();
        setTimeout(() => {
            this.updateModelDisplay();
            // if (this.currentModel === 'large') {
            //     this.selectLargeModel(this.currentLargeModel);
            // }
        }, 100);
    }

    bindEvents() {
        const sendBtn = document.getElementById('sendBtn');
        const chatInput = document.getElementById('chatInput');

        const _sendMessage = async (input, model) => {
            try {
                let reply;
                if (this.currentPlatform === 'local') {
                    // 使用本地模型
                    reply = await this.localChat.sendMessage(input, model);
                    this.addChatMessage(model, reply, 'model');
                } else if (this.currentPlatform === 'could') {
                    // 使用云端模型
                    reply = await this.couldChat.sendMessage(input, model);
                    this.addChatMessage(model, reply, 'model');
                } else {
                    this.addChatMessage('提示', "请选择本地模型或云端模型", 'system');
                }

                // 上传对话到服务器
                // this.uploadDialogToServer(input, reply);

            } catch (error) {
                console.error('发送消息失败:', error);
                this.addChatMessage('系统', `错误: ${error.message}`, 'system');
            }
        };

        sendBtn.addEventListener('click', (e) => {
            const input = chatInput.value;
            if (input) {
                this.sendMessage().then(async () => {
                    await _sendMessage(input, this.currentLargeModel);
                });
            }
            else {
                this.showMessage("请输入");
            }
        });

        chatInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    const input = chatInput.value;
                    if (input) {
                        this.sendMessage().then(async () => {
                            await _sendMessage(chatInput.value, this.currentLargeModel);
                        });
                    }
                }
            }
        );

        document.getElementById('smallModelCard').addEventListener('click', () => {
                this.currentModel = 'small';
                document.getElementById('trainingTask').classList.remove('hidden');
                document.getElementById('trainingTask').classList.add('block');
                document.getElementById('chatArea').classList.remove('block');
                document.getElementById('chatArea').classList.add('hidden');
                this.updateModelDisplay();
            }
        );

        document.getElementById('largeModelCard').addEventListener('click', () => {
            this.currentModel = 'large';
            document.getElementById('trainingTask').classList.remove('block');
            document.getElementById('trainingTask').classList.add('hidden');
            this.updateModelDisplay();
        });

        this.bindLargeModelEvents();

        // 初始化图片上传功能
        this.initImageUpload();
    }

    bindLargeModelEvents() {
        const largeModelItems = document.querySelector('#modelSelector').querySelectorAll('[data-large-model]');
        largeModelItems.forEach(item => {
                item.addEventListener('click', () => {

                    document.getElementById('chatArea').classList.remove('hidden');
                    document.getElementById('chatArea').classList.add('block');

                    const modelType = item.getAttribute('data-large-model');
                    this.currentPlatform = item.getAttribute('data-platform');

                    // 根据平台清空对应的聊天历史
                    if (this.currentPlatform === 'local') {
                        this.localChat.clearHistory();
                    } else if (this.currentPlatform === 'could') {
                        this.couldChat.clearHistory();
                    }

                    this.selectLargeModel(modelType);
                });
            }
        );
    }

    selectLargeModel(modelType) {
        this.currentLargeModel = modelType;
        const largeModelItems = document.querySelectorAll('[data-large-model]');
        largeModelItems.forEach(item => {
                const itemType = item.getAttribute('data-large-model');
                if (itemType === modelType) {
                    item.classList.add('active');
                } else {
                    item.classList.remove('active');
                }
            }
        );

        document.getElementById('chatArea').classList.remove('hidden');
        document.getElementById('chatArea').classList.add('block');
        document.getElementById('chatArea').getElementsByTagName('h2')[0].innerHTML = modelType;
    }

    bindTaskEvents() {
        const taskItems = document.querySelectorAll('.task-item');
        taskItems.forEach(item => {
                item.addEventListener('click', () => {
                        const taskId = item.getAttribute('data-task-id');
                        this.selectTask(taskId);
                    }
                );
            }
        );
    }

    updateConnectionStatus() {
        // const statusIndicator = document.getElementById('connectionStatus');
        // const connectionText = document.getElementById('connectionText');
        // statusIndicator.className = 'status-indicator status-active';
        // connectionText.textContent = '连接状态：已连接';
    }

    updateModelDisplay() {
        const smallModelCard = document.getElementById('smallModelCard');
        const largeModelCard = document.getElementById('largeModelCard');
        const largeModelList = document.getElementById('largeModelList');
        const downloadModelList = document.getElementById('downloadModelList');
        let trainingTaskSection = null;
        const sections = document.querySelectorAll('.mb-6');
        sections.forEach(section => {
                const heading = section.querySelector('h2');
                if (heading && heading.textContent.includes('训练任务')) {
                    trainingTaskSection = section;
                }
            }
        );
        if (this.currentModel === 'small') {
            smallModelCard.classList.add('active');
            largeModelCard.classList.remove('active');
            if (largeModelList) {
                largeModelList.style.display = 'none';
            }
            if (downloadModelList) {
                downloadModelList.style.display = 'none';
            }
            if (trainingTaskSection) {
                trainingTaskSection.style.display = 'block';
                trainingTaskSection.style.opacity = '0';
                setTimeout(() => {
                        trainingTaskSection.style.transition = 'opacity 0.3s ease-in-out';
                        trainingTaskSection.style.opacity = '1';
                    }
                    , 50);
            }
        } else {
            largeModelCard.classList.add('active');
            smallModelCard.classList.remove('active');
            if (largeModelList) {
                largeModelList.style.display = 'block';
                largeModelList.style.opacity = '0';
                setTimeout(() => {
                    largeModelList.style.transition = 'opacity 0.3s ease-in-out';
                    largeModelList.style.opacity = '1';
                }, 50);
            }
            if (downloadModelList && window.platform !== 'ios') {
                downloadModelList.style.display = 'block';
                downloadModelList.style.opacity = '0';
                setTimeout(() => {
                    downloadModelList.style.transition = 'opacity 0.3s ease-in-out';
                    downloadModelList.style.opacity = '1';
                }, 50);
            }
            if (this.isTraining) {
                this.stopTraining();
                this.showMessage('已切换到大模型，训练已停止');
            }
            if (trainingTaskSection) {
                trainingTaskSection.style.transition = 'opacity 0.3s ease-in-out';
                trainingTaskSection.style.opacity = '0';
                setTimeout(() => {
                        trainingTaskSection.style.display = 'none';
                    }
                    , 300);
            }
        }
    }

    stopTraining() {
        this.isTraining = false;
        const startBtn = document.getElementById('startTrainingBtn');
        const taskStatus = document.getElementById('taskStatus');
        startBtn.innerHTML = '<span class="material-icons text-sm mr-1">play_arrow</span>开始训练';
        startBtn.classList.remove('bg-red-500', 'hover:bg-red-600');
        taskStatus.className = 'status-indicator status-idle';
    }

    simulateTraining() {
        if (!this.isTraining)
            return;
        const progressIncrement = Math.random() * 5 + 2;
        this.trainingProgress = Math.min(100, this.trainingProgress + progressIncrement);
        this.updateTrainingProgress();
        if (this.trainingProgress >= 100) {
            this.completeTraining();
        } else {
            setTimeout(() => this.simulateTraining(), 500);
        }
    }

    updateTrainingProgress() {
        const progressFill = document.getElementById('progressFill');
        const progressText = document.getElementById('progressText');
        const epochCount = document.getElementById('epochCount');
        const task = this.getCurrentTask();
        progressFill.style.width = `${this.trainingProgress}%`;
        progressText.textContent = `${Math.round(this.trainingProgress)}%`;
        if (task) {
            const currentEpoch = Math.round(this.trainingProgress / 100 * task.epochs);
            epochCount.textContent = `${currentEpoch}/${task.epochs}`;
        }
    }

    completeTraining() {
        this.stopTraining();
        this.trainingCount++;
        this.showMessage('训练完成！可以上传结果到区域服务器');
        this.addChatMessage('系统', '训练任务已完成，模型参数已优化', 'system');
    }

    async sendMessage() {
        const chatInput = document.getElementById('chatInput');
        const message = chatInput.value.trim();
        if (!message) return;

        chatInput.value = '';
        this.addChatMessage('Me', message, 'user');
        this.dialogCount++;
    }

    addChatMessage(sender, message, type) {
        const chatMessages = document.getElementById('chatMessages');
        const messageDiv = document.createElement('div');
        const alignment = type === 'user' ? 'justify-end' : 'justify-start';
        const bgColor = type === 'user' ? 'bg-blue-500 text-white' : type === 'system' ? 'bg-gray-200 text-gray-700' : 'bg-white border border-gray-200';

        const html = marked.parse(message);

        messageDiv.className = `flex ${alignment}`;
        messageDiv.innerHTML = `
            <div class="chat-bubble ${bgColor} rounded-lg p-3">
                <div class="text-xs font-medium mb-1 ${type === 'user' ? 'text-blue-100' : 'text-gray-500'}">${sender}</div>
                <div class="text-sm">${html}</div>
            </div>
        `;
        chatMessages.appendChild(messageDiv);
        const chatContainer = document.getElementById('chatContainer');
        chatContainer.scrollTop = chatContainer.scrollHeight;
        this.chatHistory.push({
            sender,
            message,
            type,
            timestamp: new Date().toISOString()
        });
    }

    uploadDialogToServer(userMessage, modelResponse) {
        setTimeout(() => {
                this.uploadCount++;
                console.log('对话内容已上传到服务器:', {
                    userMessage,
                    modelResponse
                });
            }
            , 1000);
    }

    uploadDialogHistory() {
        if (this.chatHistory.length === 0) {
            this.showMessage('暂无对话记录可上传');
            return;
        }
        const uploadBtn = document.getElementById('uploadDialogBtn');
        const originalContent = uploadBtn.innerHTML;
        uploadBtn.disabled = true;
        uploadBtn.innerHTML = '<span class="material-icons text-sm mr-1 animate-spin">refresh</span>上传中...';
        setTimeout(() => {
                const dialogData = {
                    modelType: this.currentModel,
                    largeModel: this.currentLargeModel,
                    timestamp: new Date().toISOString(),
                    chatHistory: this.chatHistory.filter(item => item.type !== 'system'),
                    totalMessages: this.chatHistory.length
                };
                console.log('上传对话历史到服务器:', dialogData);
                this.uploadCount++;
                uploadBtn.innerHTML = originalContent;
                uploadBtn.disabled = false;
                this.showMessage(`成功上传 ${dialogData.totalMessages} 条对话记录到服务器`);
                this.addChatMessage('系统', '对话历史已成功上传到区域服务器和中央服务器', 'system');
            }
            , 2000);
    }

    showMessage(text, timeout = 2000) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'fixed top-4 left-1/2 transform -translate-x-1/2 bg-gray-800 text-white px-4 py-2 rounded-lg z-50';
        messageDiv.id = 'custom-message';
        messageDiv.textContent = text;
        document.body.appendChild(messageDiv);
        this.customMessageTimeout = setTimeout(() => {
            messageDiv.remove();
        }, timeout);
    }

    hideMessage() {
        clearTimeout(this.customMessageTimeout);
        this.customMessageTimeout = undefined;
        document.getElementById('custom-message').remove();
    }

    selectTask(taskId) {
        if (this.isTraining) {
            this.showMessage('训练进行中，无法切换任务');
            return;
        }
        this.currentTaskId = taskId;
        this.trainingProgress = 0;
        // this.updateTaskDisplay();
        this.updateTaskSelection();
        this.saveCurrentTask();
        this.showMessage(`已切换到: ${this.getCurrentTask().name}`);
    }

    getCurrentTask() {
        return this.tasks.find(task => task.id === this.currentTaskId);
    }

    // updateTaskDisplay() {
    //     const task = this.getCurrentTask();
    //     if (!task)
    //         return;
    //     document.getElementById('taskName').textContent = task.name;
    //     document.getElementById('taskDescription').textContent = task.description;
    //     document.getElementById('sampleCount').textContent = task.sampleCount.toLocaleString();
    //     document.getElementById('epochCount').textContent = `0/${task.epochs}`;
    //     document.getElementById('progressText').textContent = '0%';
    //     document.getElementById('progressFill').style.width = '0%';
    // }

    updateTaskSelection() {
        const taskItems = document.querySelectorAll('.task-item');
        taskItems.forEach(item => {
                const taskId = item.getAttribute('data-task-id');
                if (taskId === this.currentTaskId) {
                    item.classList.add('selected');
                } else {
                    item.classList.remove('selected');
                }
            }
        );
    }

    refreshTasks() {
        const refreshBtn = document.getElementById('refreshTasksBtn');
        const originalContent = refreshBtn.innerHTML;
        refreshBtn.innerHTML = '<span class="material-icons text-sm align-middle mr-1 animate-spin">refresh</span>刷新中...';
        refreshBtn.disabled = true;
        setTimeout(() => {
                this.showMessage('任务列表已刷新');
                refreshBtn.innerHTML = originalContent;
                refreshBtn.disabled = false;
            }
            , 1500);
    }

    saveCurrentTask() {
        const stats = {
            trainingCount: this.trainingCount,
            dialogCount: this.dialogCount,
            uploadCount: this.uploadCount,
            currentTaskId: this.currentTaskId,
            currentModel: this.currentModel,
            currentLargeModel: this.currentLargeModel,
            samplePath: this.samplePath
        };
        localStorage.setItem('federatedLearningStats', JSON.stringify(stats));
    }

    /**
     * 初始化图片上传功能
     */
    initImageUpload() {
        const fileBtn = document.getElementById('fileBtn');
        const imageUpload = document.getElementById('imageUpload');

        // 点击文件按钮触发文件选择
        fileBtn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            imageUpload.click();
        });

        // 文件选择变化事件
        imageUpload.addEventListener('change', async (e) => {
            e.stopPropagation();
            const files = e.target.files;
            if (files.length > 0) {
                await this.handleImageUpload(files[0]);
            }
            // 清空文件输入，允许选择同一文件
            imageUpload.value = '';
        });

        // 阻止文件输入的点击事件冒泡
        imageUpload.addEventListener('click', (e) => {
            e.stopPropagation();
        });
    }

    /**
     * 处理图片上传
     * @param {File} imageFile - 图片文件
     */
    async handleImageUpload(imageFile) {
        try {
            // 检查文件类型
            if (!imageFile.type.startsWith('image/')) {
                this.showMessage('请选择图片文件');
                return;
            }

            // 检查文件大小（限制为5MB）
            if (imageFile.size > 5 * 1024 * 1024) {
                this.showMessage('图片大小不能超过5MB');
                return;
            }

            // 显示上传提示
            this.showMessage('正在处理图片...', 30000);

            // 转换图片为base64
            const base64Image = await this.convertImageToBase64(imageFile);

            // 在聊天界面显示图片
            this.addImageMessage('Me', base64Image, '上传的图片', 'user');

            // 根据平台处理图片
            let reply;
            if (this.currentPlatform === 'could') {
                reply = await this.sendImageToCloudAPI(base64Image);
            } else if (this.currentPlatform === 'local') {
                reply = await this.sendImageToLocalAPI(base64Image);
            } else {
                this.addChatMessage('提示', "请先选择模型平台", 'system');
                return;
            }

            // 隐藏消息
            this.hideMessage();

            // 添加助手回复
            this.addChatMessage(this.currentLargeModel, reply, 'model');

        } catch (error) {
            console.error('图片处理失败:', error);
            this.hideMessage();
            this.addChatMessage('系统', `图片处理错误: ${error.message}`, 'system');
        }
    }

    /**
     * 将图片转换为base64格式
     * @param {File} file - 图片文件
     * @returns {Promise<string>} - base64字符串
     */
    convertImageToBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                resolve(e.target.result);
            };
            reader.onerror = (error) => {
                reject(new Error('读取图片文件失败'));
            };
            reader.readAsDataURL(file);
        });
    }

    /**
     * 发送图片到云端API
     * @param {string} base64Image - base64格式的图片
     * @returns {Promise<string>} - 助手回复
     */
    async sendImageToCloudAPI(base64Image) {
        try {
            // 创建包含图片的消息
            const imageMessage = {
                role: 'user',
                content: [
                    {
                        type: 'text',
                        text: '请描述这张图片的内容'
                    },
                    {
                        type: 'image_url',
                        image_url: {
                            url: base64Image
                        }
                    }
                ]
            };

            // 添加到消息历史
            this.couldChat.messages.push(imageMessage);

            // 准备请求数据（使用支持图片的模型）
            const requestData = {
                model: 'gpt-4-vision-preview', // 使用支持视觉的模型
                messages: this.couldChat.messages,
                max_tokens: 1000,
                stream: false
            };

            // 发送请求
            const response = await this.couldChat.client.post('/chat/completions', requestData);

            // 获取回复
            const assistantReply = response.data.choices[0].message.content;

            // 添加到历史
            this.couldChat.messages.push({
                role: 'assistant',
                content: assistantReply
            });

            return assistantReply;

        } catch (error) {
            throw new Error(`云端图片处理失败: ${error.message}`);
        }
    }

    /**
     * 发送图片到本地API
     * @param {string} base64Image - base64格式的图片
     * @returns {Promise<string>} - 助手回复
     */
    async sendImageToLocalAPI(base64Image) {
        try {
            // 移除base64前缀
            const base64Data = base64Image.split(',')[1];

            // 准备包含图片的消息
            const imageMessage = {
                role: 'user',
                content: '请描述这张图片的内容',
                images: [base64Data]
            };

            // 添加到消息历史
            this.localChat.messages.push(imageMessage);

            // 准备请求数据（Ollama可能需要特定的图片处理方式）
            const requestData = {
                model: this.currentLargeModel,
                messages: this.localChat.messages,
                stream: false
            };

            // 发送请求（注意：Ollama可能需要不同的端点或格式）
            this.localChat.client.baseURL = window.ollamaHost;
            const response = await this.localChat.client.post('/api/chat', requestData);

            // 获取回复
            const assistantReply = response.data.message.content;

            // 添加到历史
            this.localChat.messages.push({
                role: 'assistant',
                content: assistantReply
            });

            return assistantReply;

        } catch (error) {
            // 如果本地模型不支持图片，则尝试其他方式
            if (error.response && error.response.status === 400) {
                throw new Error('当前本地模型可能不支持图片处理');
            }
            throw new Error(`本地图片处理失败: ${error.message}`);
        }
    }

    /**
     * 添加图片消息到聊天界面
     * @param {string} sender - 发送者
     * @param {string} imageUrl - 图片URL
     * @param {string} message - 附加消息
     * @param {string} type - 消息类型
     */
    addImageMessage(sender, imageUrl, message = '', type = 'user') {
        const chatMessages = document.getElementById('chatMessages');
        const messageDiv = document.createElement('div');
        const alignment = type === 'user' ? 'justify-end' : 'justify-start';
        const bgColor = type === 'user' ? 'bg-blue-500 text-white' : 'bg-white border border-gray-200';

        messageDiv.className = `flex ${alignment} mb-3`;
        messageDiv.innerHTML = `
            <div class="chat-bubble ${bgColor} rounded-lg p-3 max-w-xs">
                <div class="text-xs font-medium mb-1 ${type === 'user' ? 'text-blue-100' : 'text-gray-500'}">${sender}</div>
                ${message ? `<div class="text-sm mb-2">${message}</div>` : ''}
                <div class="image-container">
                    <img src="${imageUrl}" alt="上传的图片" class="max-w-full h-auto rounded">
                </div>
            </div>
        `;
        chatMessages.appendChild(messageDiv);

        // 滚动到底部
        const chatContainer = document.getElementById('chatContainer');
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
}

// 模型管理器类
class ModelManager {
    constructor() {
        this.models = [];
        this.selectedModel = null;
    }

    // 创建模型卡片HTML
    createModelCardHTML(model) {
        const isSelected = this.selectedModel === model.id;
        return `
            <div class="model-card col-span-1 rounded-lg p-3 cursor-pointer border transition-all duration-200
                        ${isSelected ? 'bg-blue-100 border-blue-500' : 'border-gray-200 hover:border-blue-500 hover:bg-blue-50'}"
                 data-large-model="${model.id}"
                 data-platform="${model.platform}"
                 data-testid="large-model-${model.id}">
                <div class="flex items-center justify-between">
                    <div>
                        <div class="font-medium text-sm text-gray-800">${model.name}</div>
                        ${model.description ? `<div class="text-xs text-gray-500 mt-1">${model.description}</div>` : ''}
                        <div class="text-xs mt-1 ${model.platform === 'local' ? 'text-green-600' : 'text-blue-600'}">
                            ${model.platform === 'local' ? 'Local Model' : 'Could Model'}
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    // 渲染模型列表
    render(containerId = 'modelListContainer') {
        const container = document.getElementById(containerId);
        if (!container) return;

        container.innerHTML = this.models.map(model =>
            this.createModelCardHTML(model)
        ).join('');
    }

    // 选择模型
    selectModel(modelId) {
        this.selectedModel = modelId;
        this.render();

        const event = new CustomEvent('modelSelected', {
            detail: { modelId, model: this.models.find(m => m.id === modelId) }
        });
        document.dispatchEvent(event);
    }

    // 获取选中的模型
    getSelectedModel() {
        return this.models.find(model => model.id === this.selectedModel);
    }

    // 添加新模型
    addModel(model) {
        this.models.push(model);
        this.render();
    }

    // 移除模型
    removeModel(modelId) {
        this.models = this.models.filter(model => model.id !== modelId);
        this.render();
    }

    async getModelList() {
        const config = {
            method: 'get',
            url: 'https://api.chatanywhere.tech/v1/models',
            headers: {
                'Authorization': 'Bearer sk-sjvqTxWynT1Y5isDZ0Rs78eIGLsSCSLYlWbEKxNGXOuAuWsF'
            }
        };

        return axios(config);
    }

    async getLocalModalList() {
        const config = {
            method: 'get',
            url: `http://${window.ollamaHost}/api/tags`,
            withCredentials: false,
        };

        return axios(config);
    }

    async syncSelfCouldModels() {
        const config = {
            method: 'get',
            url: `http://${window.serverHost}/api/v1/learn_management/model_info/`,
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${window.authToken}`,
            },
        };

        return axios(config);
    }

    async getSelfCouldModelVersion() {
        const config = {
            method: 'get',
            url: `http://${window.serverHost}/api/v1/learn_management/model_version/`,
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${window.authToken}`,
            },
        };

        return axios(config);
    }
}

const fla = new FederatedLearningApp();
const modelManager = new ModelManager();

// 合并云端模型
function deduplicateModels(models) {
    const seen = new Set();
    const result = [];

    models.forEach(model => {
        const key = `${model.id}|${model.owned_by}`;
        if (!seen.has(key)) {
            seen.add(key);
            result.push(model);
        }
    });

    return result;
}

// 设置对话模型时间
function setLargeModelCardListener() {
    document.getElementById('largeModelCard').addEventListener('click', () => {
        // 清空模型列表
        modelManager.models = [];

        // 检查服务器地址和端口
        if (!window.serverHost) {
            fla.showMessage("Please open settings and set the server host.");
            document.querySelector('#eventSettings').click();
            return;
        }

        // 获取自有云端模型
        if (window.platform !== 'ios') {
            modelManager.syncSelfCouldModels()
                .then((response) => {
                    const { data } = response.data;
                    let allCardHTML = '';
                    data.list.forEach(model => {
                        const cardHTML = modelManager.createModelCardHTML({
                            id: model.id,
                            name: model.name,
                            platform: 'could',
                            description: model.description,
                        });
                        allCardHTML += cardHTML;
                    });
                    document.getElementById('downloadModelListContainer').innerHTML = allCardHTML;
                    // 获取模板版本列表
                    const modelVersion = document.getElementById('sync-model-version-modal');
                    // 模型点击事件
                    const modelCards = document.getElementById('downloadModelListContainer').querySelectorAll('.model-card')
                    modelCards.forEach(element => {
                        element.addEventListener('click', (e) => {
                            modelVersion.classList.remove('hidden');
                            modelVersion.classList.add('block');
                            const id = e.currentTarget.getAttribute('data-large-model');
                            const name = e.currentTarget.querySelector('.font-medium').innerHTML;
                            modelManager.getSelfCouldModelVersion().then((response) => {
                                const  { data } = response.data;
                                const versions = data.list.filter(v => v.model_info === parseInt(id));
                                const group = modelVersion.querySelector('.group');
                                let html = '';
                                versions.forEach(v => {
                                    html += '<div class="item flex justify-between">\n' +
                                        '<span class="text-sm text-gray-600">' + v.version + '</span>\n' +
                                        '<span class="text-sm text-blue-600" data-model-file="' + v.model_file + '" data-id="' + v.id +'">Click Sync</span>\n' +
                                        '</div>';
                                });
                                group.innerHTML = html;
                                group.querySelectorAll('.item').forEach(item => {
                                    item.querySelector('[data-id]').onclick = (e) => {
                                        const item_id = e.currentTarget.getAttribute('data-id')
                                        const item_model_file = e.currentTarget.getAttribute('data-model-file');
                                        if (typeof Android !== 'undefined') {
                                            fla.showMessage("Syncing model...", 60 * 1000);
                                            Android.syncModel(`http://${window.serverHost}/api/v1`, window.authToken, name, item_id, item_model_file);
                                        }
                                        else {
                                            fla.showMessage('环境不匹配，请在TERMUX中执行');
                                        }
                                    }
                                });
                            }).catch((reason) => {
                                fla.showMessage("获取自建云模型版本列表失败：" + reason.message);
                            });
                        });
                    });
                    modelVersion.querySelector('.close').addEventListener('click', (e) => {
                        modelVersion.classList.remove('block');
                        modelVersion.classList.add('hidden');
                    })
                })
                .catch((error) => {
                    fla.showMessage(error.message);
                });
        }

        // 读取远程云端模型
        modelManager.getModelList()
            .then(function (response) {
                const { data } = response.data;
                deduplicateModels(data).forEach((model) => {
                    modelManager.addModel({
                        id: model.id,
                        name: model.id,
                        description: model.owned_by,
                        platform: 'could',
                    });
                })
                if (fla) {
                    fla.bindLargeModelEvents();
                }
            })
            .catch(function (error) {
                console.log('获取云端模型失败:', error.toString());
            })

        if (window.platform !== 'ios') {
            // 读取本地模型
            modelManager.getLocalModalList()
                .then(function (response) {
                    const { models } = response.data;
                    models.forEach(model => {
                        modelManager.addModel({
                            id: model.model,
                            name: model.model,
                            description: `Local ${model.details.parameter_size} Model`,
                            platform: 'local',
                        });
                    })
                    if (fla) {
                        fla.bindLargeModelEvents();
                    }
                })
                .catch(function (error) {
                    console.log('获取本地模型失败:', error.toString());
                })
        }
    })
}

// 设置保存对话事件
function setSaveChatListener() {
    const save = document.getElementById('saveBtn');
    save.addEventListener('click', () => {
        const modal = document.getElementById('save-chat-history-modal');
        modal.classList.remove('hidden');
        modal.classList.add('block');
    });
    const saveFinish = document.getElementById('save-finish');
    saveFinish.addEventListener('click', () => {
        const modelVersionInput = document.querySelector('input[name="model_version"]');
        const edgeNodeInput = document.querySelector('input[name="edge_node"]');

        if (!modelVersionInput.value || !edgeNodeInput.value) {
            fla.showMessage("请填写配置！");
            return;
        }

        const modal = document.getElementById('save-chat-history-modal');
        modal.classList.remove('block');
        modal.classList.add('hidden');
        if (modelVersionInput.value && edgeNodeInput.value && fla && fla.currentPlatform === 'local') {
            const data=  {
                "created_by_detail": {
                    "name": "user",
                    "email": "",
                    "is_active": true,
                    "is_superuser": true
                },
                "input_data": {
                    "role": "",
                    "content": "",
                    "timestamp": "",
                    "model": "",
                },
                "output_data": {
                    "role": "",
                    "content": "",
                    "timestamp": "",
                },
                "error_message": "",
                "model_version": modelVersionInput.value,
                "edge_node": edgeNodeInput.value,
            };
            if (fla.localChat.messages.length > 0) {
                fla.localChat.messages.forEach(item => {
                    if (item['role'] === 'user') {
                        data['input_data']['role'] = item['role'];
                        data['input_data']['content'] = item['content'];
                        data['input_data']['timestamp'] = new Date().getTime();
                        data['input_data']['model'] = fla.currentLargeModel;
                    }
                    else if (item['role'] === 'assistant') {
                        data['output_data']['role'] = item['role'];
                        data['output_data']['content'] = item['content'];
                        data['output_data']['timestamp'] = new Date().getTime();
                    }
                });
                const config = {
                    method: 'post',
                    url: `http://${window.serverHost}/api/v1/learn_management/model_inference/`,
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${window.authToken}`,
                    },
                    data: data,
                };
                axios(config)
                    .then(response => {
                        const { id } = response.data;
                        if (id) {
                            fla.showMessage("对话已保存");
                        }
                    })
                    .catch((reason) => {
                        fla.showMessage(reason.message);
                    });
            }
            else {
                fla.showMessage("暂无历史记录！");
            }
        }
        else {
            fla.showMessage("对话保存失败，请检查配置！");
        }
    })
    const saveCancel = document.getElementById('save-cancel');
    saveCancel.addEventListener('click', () => {
        const modal = document.getElementById('save-chat-history-modal');
        modal.classList.remove('block');
        modal.classList.add('hidden');
    });
}

document.addEventListener('DOMContentLoaded', () => {
    // 初始化渲染
    modelManager.render();

    // 显示容器
    document.getElementById('largeModelList').style.display = 'block';

    if (fla) {
        fla.init();
        fla.bindEvents();
    }

    // 点击对话模型卡片事件
    setLargeModelCardListener();

    if (window.platform === 'ios') {
        document.getElementById('status-bar').querySelector('.flex').style.display = 'none';
        document.getElementById('settings').querySelectorAll('.item')[0].style.display = 'none';
    }

    // 保存对话
    if (window.platform !== 'ios') {
        setSaveChatListener();
    }

    if (window.platform === 'ios') {
        // small
        document.getElementById('smallModelCard').classList.remove('block');
        document.getElementById('smallModelCard').classList.remove('active');
        document.getElementById('smallModelCard').classList.add('hidden');

        setTimeout(() => {
            document.getElementById('saveBtn').style.display = 'none';
            document.getElementById('fileBtn').style.right = '46px';
            document.getElementById('largeModelCard').click();
        }, 100);
    }
});

async function login(data) {
    const config = {
        method: 'post',
        url: `http://${window.serverHost}/api/v1/account/login/`,
        headers: {
            'Content-Type': 'application/json',
        },
        data,
    };

    return axios(config);
}

document.addEventListener('DOMContentLoaded', () => {
    const settings = document.querySelector('#settings');
    const actions = settings.querySelector('.actions');

    // close setting modal
    actions.querySelector('.btn-back').addEventListener('click', () => {
        settings.classList.add('hidden');
        settings.classList.remove('block');
    });

    // open setting modal
    const eventSettings = document.querySelector('#eventSettings');
    eventSettings.addEventListener('click', () => {
        settings.classList.add('block');
        settings.classList.remove('hidden');
    });

    // initialize setting value
    const ollamaHost = localStorage.getItem('ollamaHost');
    const serverHost = localStorage.getItem('serverHost');
    const serverUserName = localStorage.getItem('serverUserName');
    if (ollamaHost) {
        window.ollamaHost = ollamaHost;
        settings.querySelector('input[name=ollamaHost]').value = ollamaHost;
    }
    if (serverHost) {
        window.serverHost = serverHost;
        settings.querySelector('input[name=serverHost]').value = serverHost;
    }
    if (serverUserName) {
        window.serverUserName = serverUserName;
        settings.querySelector('input[name=serverUserName]').value = serverUserName;
    }
    window.authToken = localStorage.getItem('authToken');

    // save setting modal
    actions.querySelector('.btn-save').addEventListener('click', () => {
        settings.classList.add('hidden');
        settings.classList.remove('block');

        window.ollamaHost = settings.querySelector('input[name=ollamaHost]').value;
        window.serverHost = settings.querySelector('input[name=serverHost]').value;
        window.serverUserName = settings.querySelector('input[name=serverUserName]').value;
        window.serverPassword = settings.querySelector('input[name=serverPassword]').value;
        localStorage.setItem('ollamaHost', window.ollamaHost);
        localStorage.setItem('serverHost', window.serverHost);
        localStorage.setItem('serverUserName', window.serverUserName);

        // login
        const token = localStorage.getItem('authToken');
        if (token) {
            return
        }
        login({ name: window.serverUserName, password: window.serverPassword }).then((response) => {
            const { access } = response.data;
            window.authToken = access;
            localStorage.setItem('authToken', access);
            document.getElementById('largeModelCard').click();
        }).catch((reason) => {
            console.log(reason.message);
        });
    });
});

