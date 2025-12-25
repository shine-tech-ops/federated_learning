if (!window._candyJarCallbacks) {
    window._candyJarCallbacks = {};
}
if (!window.CandyJar) {
    window.CandyJar = {
        call: function(method, action, params, successCallback, errorCallback, timeout=10000) {
            const callId = 'candyJar_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
            if (!window._candyJarCallbacks) {
                window._candyJarCallbacks = {};
            }
            window._candyJarCallbacks[callId] = {
                resolve: (result) => {
                    clearTimeout(timeoutId);
                    delete window._candyJarCallbacks[callId];
                    if (successCallback && typeof successCallback === 'function') {
                        successCallback(result);
                    }
                }
                ,
                reject: (error) => {
                    clearTimeout(timeoutId);
                    delete window._candyJarCallbacks[callId];
                    if (errorCallback && typeof errorCallback === 'function') {
                        errorCallback(error);
                    }
                }
            };
            const timeoutId = setTimeout( () => {
                    if (window._candyJarCallbacks[callId]) {
                        console.warn('CandyJar 调用超时:', method, action);
                        delete window._candyJarCallbacks[callId];
                        if (errorCallback && typeof errorCallback === 'function') {
                            errorCallback(new Error('CandyJar call timeout'));
                        }
                    }
                }
                , timeout);
            const message = {
                type: 'CANDYJAR_CALL',
                callId: callId,
                method: method,
                action: action,
                params: params
            };
            console.log('子iframe向主iframe发送消息:', JSON.stringify(message, null, 2));
            window.parent.postMessage(message, '*');
        }
    };
    window.addEventListener('message', function(event) {
        if (event.data && event.data.type === 'CANDYJAR_CALLBACK') {
            const {callId, success, result, error} = event.data;
            if (window._candyJarCallbacks && window._candyJarCallbacks[callId]) {
                const promiseData = window._candyJarCallbacks[callId];
                if (success) {
                    console.log('CandyJar 调用成功:', callId, result);
                    promiseData.resolve(result);
                } else {
                    console.error('CandyJar 调用失败:', callId, error);
                    promiseData.reject(new Error(error || 'CandyJar call failed'));
                }
            } else {
                console.warn('未找到对应的回调处理器（可能已经释放） for callId:', callId);
            }
        }
        if (event.data && event.data.type === 'CANDYJAR_EVENT') {
            const {eventType, eventData} = event.data;
            console.log('子iframe 接收到' + eventType + '类型的事件转发，数据:' + JSON.stringify(eventData, null, 2));
            const customEvent = new CustomEvent(eventType,{
                detail: eventData,
                bubbles: true,
                cancelable: true
            });
            if (eventData.segmentId !== undefined)
                customEvent.segmentId = eventData.segmentId;
            if (eventData.sentences !== undefined)
                customEvent.sentences = eventData.sentences;
            if (eventData.error !== undefined)
                customEvent.error = eventData.error;
            if (eventData.htmlFragment !== undefined)
                customEvent.htmlFragment = eventData.htmlFragment;
            if (eventData.action !== undefined)
                customEvent.action = eventData.action;
            console.log('iframe - 最终构建的customEvent:', JSON.stringify(customEvent, null, 2));
            window.document.dispatchEvent(customEvent);
        }
    });
    console.log('iframe CandyJar 已初始化为 postMessage 通信');
}
if (document.body.style.overflow === 'hidden') {
    document.body.style.removeProperty('overflow');
}
(function() {
        class AudioContext2 {
            constructor() {
                this._startedAtMs = performance.now();
                this.destination = {
                    _isDestination: true,
                    connect: function() {}
                };
                this.nextId = 1;
                this.sampleRate = 44100;
            }
            get currentTime() {
                return (performance.now() - this._startedAtMs) / 1000;
            }
            _postMessage(payload) {
                try {
                    const callId = 'candyJar_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
                    const audioEvent = {
                        type: 'CANDYJAR_CALL',
                        callId: callId,
                        method: "AudioContext2",
                        action: "event",
                        params: payload
                    }
                    window.parent.postMessage(audioEvent, '*');
                    console.log('AudioContext2: postMessage success: ', JSON.stringify(audioEvent, null, 2));
                    return callId;
                } catch (e) {
                    console.warn('AudioContext2: postMessage error', e);
                    return null;
                }
            }
            _sendDelayedMessage(payload, delayMs) {
                const MAX_DELAY = 30000;
                const actualDelay = Math.min(delayMs, MAX_DELAY);
                if (actualDelay < delayMs) {
                    console.warn('AudioContext2: delay capped at', MAX_DELAY, 'ms (requested:', delayMs + 'ms)');
                }
                if (actualDelay > 0) {
                    setTimeout( () => {
                            this._postMessage(payload);
                        }
                        , actualDelay);
                } else {
                    this._postMessage(payload);
                }
            }
            createOscillator() {
                const id = this.nextId++;
                const ctx = this;
                const oscillator = {
                    id: id,
                    frequency: {
                        value: 440,
                        _automation: [],
                        setValueAtTime: function(value, time) {
                            this.value = value;
                            this._automation.push({
                                type: 'set',
                                value,
                                time
                            });
                        },
                        linearRampToValueAtTime: function(value, time) {
                            this._automation.push({
                                type: 'linear',
                                value,
                                time
                            });
                        },
                        exponentialRampToValueAtTime: function(value, time) {
                            this._automation.push({
                                type: 'exp',
                                value,
                                time
                            });
                        },
                    },
                    type: 'sine',
                    _connectedToDestination: false,
                    _started: false,
                    _stopped: false,
                    _startAt: null,
                    _stopAt: null,
                    _gainChain: [],
                    _instanceId: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
                    connect: function(destination) {
                        if (!destination)
                            return this;
                        if (destination._isGainNode) {
                            destination._upstreamOscillators.push(this);
                            if (!this._gainChain.includes(destination)) {
                                this._gainChain.push(destination);
                            }
                        }
                        if (destination._isDestination) {
                            this._connectedToDestination = true;
                        }
                        return this;
                    },
                    disconnect: function(destination) {
                        if (!destination) {
                            this._connectedToDestination = false;
                            this._gainChain = [];
                            return this;
                        }
                        if (destination._isGainNode) {
                            const index = destination._upstreamOscillators.indexOf(this);
                            if (index > -1) {
                                destination._upstreamOscillators.splice(index, 1);
                            }
                            const gainIndex = this._gainChain.indexOf(destination);
                            if (gainIndex > -1) {
                                this._gainChain.splice(gainIndex, 1);
                            }
                        }
                        if (destination._isDestination) {
                            this._connectedToDestination = false;
                        }
                        return this;
                    },
                    start: function(when) {
                        if (this._started)
                            return;
                        this._started = true;
                        const now = ctx.currentTime;
                        const startAt = (typeof when === 'number' ? when : now);
                        this._startAt = startAt;
                        if (!this._connectedToDestination) {
                            console.warn('AudioContext2: oscillator not connected to destination, ignore start');
                            return;
                        }
                        this._startPromise = new Promise( (resolve, reject) => {
                                const timeout = setTimeout( () => {
                                        reject(new Error('start timeout'));
                                    }
                                    , 1000);
                                this._startResolve = (result) => {
                                    clearTimeout(timeout);
                                    resolve(result);
                                }
                                ;
                                this._startReject = (error) => {
                                    clearTimeout(timeout);
                                    reject(error);
                                }
                                ;
                            }
                        );
                        const SILENT_THRESHOLD = 0.01;
                        const DEFAULT_DURATION = 30.0;
                        let latestAutomationTime = 0;
                        let latestSilentTime = 0;
                        let hasSilentEvent = false;
                        if (this.frequency._automation && Array.isArray(this.frequency._automation)) {
                            for (const ev of this.frequency._automation) {
                                if (typeof ev.time === 'number' && ev.time > latestAutomationTime) {
                                    latestAutomationTime = ev.time;
                                }
                            }
                        }
                        for (const gainNode of this._gainChain) {
                            if (gainNode && Array.isArray(gainNode._automation)) {
                                for (const ev of gainNode._automation) {
                                    if (typeof ev.time === 'number' && ev.time > latestAutomationTime) {
                                        latestAutomationTime = ev.time;
                                    }
                                    if (typeof ev.value === 'number' && ev.value <= SILENT_THRESHOLD) {
                                        hasSilentEvent = true;
                                        if (ev.time > latestSilentTime) {
                                            latestSilentTime = ev.time;
                                        }
                                    }
                                }
                            }
                        }
                        const maxAutomationTime = hasSilentEvent ? latestSilentTime : latestAutomationTime;
                        const calculatedDuration = (maxAutomationTime - startAt) || 0;
                        const expectedDurationSec = hasSilentEvent ? calculatedDuration : Math.max(DEFAULT_DURATION, calculatedDuration);
                        const expectedDurationMs = Math.round((expectedDurationSec > 0 ? expectedDurationSec : DEFAULT_DURATION) * 1000);
                        const frequencyAutomation = [];
                        if (this.frequency._automation) {
                            for (const ev of this.frequency._automation) {
                                const relativeTime = ev.time - startAt;
                                if (relativeTime >= 0) {
                                    frequencyAutomation.push({
                                        type: ev.type,
                                        value: ev.value,
                                        time: relativeTime
                                    });
                                }
                            }
                        }
                        const gainChain = [];
                        for (const gainNode of this._gainChain) {
                            if (gainNode && gainNode.gain) {
                                const automation = [];
                                if (gainNode._automation) {
                                    for (const ev of gainNode._automation) {
                                        const relativeTime = ev.time - startAt;
                                        if (relativeTime >= 0) {
                                            automation.push({
                                                type: ev.type,
                                                value: ev.value,
                                                time: relativeTime
                                            });
                                        }
                                    }
                                }
                                gainChain.push({
                                    id: gainNode.id,
                                    gain: {
                                        value: gainNode.gain.value,
                                        automation
                                    }
                                });
                            }
                        }
                        const callId = ctx._postMessage({
                            action: 'startOscillator',
                            id: this._instanceId,
                            frequency: {
                                value: this.frequency.value,
                                automation: frequencyAutomation
                            },
                            type: this.type,
                            expectedDurationMs: expectedDurationMs,
                            gainChain: gainChain
                        });
                        if (callId) {
                            this._startCallId = callId;
                            window._candyJarCallbacks[callId] = {
                                resolve: this._startResolve,
                                reject: this._startReject
                            };
                        }
                        console.log('AudioContext2: oscillator.start at', startAt.toFixed(3), '-> startOscillator', this.frequency.value, expectedDurationMs + 'ms');
                    },
                    stop: function(when) {
                        if (this._stopped)
                            return;
                        const now = ctx.currentTime;
                        this._stopAt = (typeof when === 'number' ? when : now);
                        const shouldDelay = typeof when === 'number' && when > now;
                        if (!this._connectedToDestination) {
                            console.warn('AudioContext2: oscillator not connected to destination, ignore stop');
                            return;
                        }
                        (async () => {
                                try {
                                    if (this._startPromise) {
                                        await this._startPromise;
                                    }
                                    if (shouldDelay) {
                                        const delayMs = Math.max(0, (when - now) * 1000);
                                        setTimeout( () => {
                                                ctx._postMessage({
                                                    action: 'stopOscillator',
                                                    id: this._instanceId
                                                });
                                                this._stopped = true;
                                                console.log('AudioContext2: 延迟停止 at', when.toFixed(3), '(delay:', delayMs + 'ms)');
                                            }
                                            , delayMs);
                                    } else {
                                        ctx._postMessage({
                                            action: 'stopOscillator',
                                            id: this._instanceId
                                        });
                                        this._stopped = true;
                                        console.log('AudioContext2: 立即停止 at', now.toFixed(3));
                                    }
                                    if (this._startCallId) {
                                        delete window._candyJarCallbacks[this._startCallId];
                                    }
                                } catch (error) {
                                    console.warn('AudioContext2: start failed, skip stop for', this._instanceId, error);
                                    if (this._startCallId) {
                                        delete window._candyJarCallbacks[this._startCallId];
                                    }
                                }
                            }
                        )();
                    }
                };
                return oscillator;
            }
            createGain() {
                const id = this.nextId++;
                const gainNode = {
                    id: id,
                    _isGainNode: true,
                    _upstreamOscillators: [],
                    _automation: [],
                    gain: {
                        value: 1,
                        setValueAtTime: function(value, time) {
                            gainNode._automation.push({
                                type: 'set',
                                value,
                                time
                            });
                        },
                        linearRampToValueAtTime: function(value, time) {
                            gainNode._automation.push({
                                type: 'linear',
                                value,
                                time
                            });
                        },
                        exponentialRampToValueAtTime: function(value, time) {
                            gainNode._automation.push({
                                type: 'exp',
                                value,
                                time
                            });
                        },
                        cancelScheduledValues: function(time) {
                            gainNode._automation = gainNode._automation.filter(e => !(typeof e.time === 'number' && e.time >= time));
                        }
                    },
                    connect: function(destination) {
                        if (!destination)
                            return this;
                        if (destination._isDestination) {
                            this._upstreamOscillators.forEach(osc => {
                                    osc._connectedToDestination = true;
                                    if (!osc._gainChain.includes(this)) {
                                        osc._gainChain.push(this);
                                    }
                                }
                            );
                        } else if (destination._isGainNode) {
                            const set = new Set(destination._upstreamOscillators);
                            this._upstreamOscillators.forEach(osc => set.add(osc));
                            destination._upstreamOscillators = Array.from(set);
                            this._upstreamOscillators.forEach(osc => {
                                    if (!osc._gainChain.includes(destination)) {
                                        osc._gainChain.push(destination);
                                    }
                                }
                            );
                        }
                        return this;
                    },
                    disconnect: function(destination) {
                        if (!destination) {
                            this._upstreamOscillators.forEach(osc => {
                                    osc._connectedToDestination = false;
                                    const index = osc._gainChain.indexOf(this);
                                    if (index > -1) {
                                        osc._gainChain.splice(index, 1);
                                    }
                                }
                            );
                            this._upstreamOscillators = [];
                            return this;
                        }
                        if (destination._isDestination) {
                            this._upstreamOscillators.forEach(osc => {
                                    osc._connectedToDestination = false;
                                    const index = osc._gainChain.indexOf(this);
                                    if (index > -1) {
                                        osc._gainChain.splice(index, 1);
                                    }
                                }
                            );
                        } else if (destination._isGainNode) {
                            const index = destination._upstreamOscillators.indexOf(this);
                            if (index > -1) {
                                destination._upstreamOscillators.splice(index, 1);
                            }
                            this._upstreamOscillators.forEach(osc => {
                                    const gainIndex = osc._gainChain.indexOf(destination);
                                    if (gainIndex > -1) {
                                        osc._gainChain.splice(gainIndex, 1);
                                    }
                                }
                            );
                        }
                        return this;
                    }
                };
                return gainNode;
            }
            createBufferSource() {
                const id = this.nextId++;
                const ctx = this;
                const bufferSource = {
                    id: id,
                    buffer: null,
                    loop: false,
                    loopStart: 0,
                    loopEnd: 0,
                    playbackRate: {
                        value: 1.0
                    },
                    _connectedToDestination: false,
                    _started: false,
                    _stopped: false,
                    _startAt: null,
                    _stopAt: null,
                    _gainChain: [],
                    _instanceId: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
                    connect: function(destination) {
                        if (!destination)
                            return this;
                        if (destination._isGainNode) {
                            destination._upstreamOscillators.push(this);
                            if (!this._gainChain.includes(destination)) {
                                this._gainChain.push(destination);
                            }
                        }
                        if (destination._isDestination) {
                            this._connectedToDestination = true;
                        }
                        return this;
                    },
                    disconnect: function(destination) {
                        if (!destination) {
                            this._connectedToDestination = false;
                            this._gainChain = [];
                            return this;
                        }
                        if (destination._isGainNode) {
                            const index = destination._upstreamOscillators.indexOf(this);
                            if (index > -1) {
                                destination._upstreamOscillators.splice(index, 1);
                            }
                            const gainIndex = this._gainChain.indexOf(destination);
                            if (gainIndex > -1) {
                                this._gainChain.splice(gainIndex, 1);
                            }
                        }
                        if (destination._isDestination) {
                            this._connectedToDestination = false;
                        }
                        return this;
                    },
                    start: function(when, offset, duration) {
                        if (this._started)
                            return;
                        this._started = true;
                        const now = ctx.currentTime;
                        const startAt = (typeof when === 'number' ? when : now);
                        this._startAt = startAt;
                        if (!this._connectedToDestination) {
                            console.warn('AudioContext2: bufferSource not connected to destination, ignore start');
                            return;
                        }
                        if (!this.buffer) {
                            console.warn('AudioContext2: bufferSource has no buffer, ignore start');
                            return;
                        }
                        const gainChain = [];
                        for (const gainNode of this._gainChain) {
                            if (gainNode && gainNode.gain) {
                                const automation = [];
                                if (gainNode._automation) {
                                    for (const ev of gainNode._automation) {
                                        const relativeTime = ev.time - startAt;
                                        if (relativeTime >= 0) {
                                            automation.push({
                                                type: ev.type,
                                                value: ev.value,
                                                time: relativeTime
                                            });
                                        }
                                    }
                                }
                                gainChain.push({
                                    id: gainNode.id,
                                    gain: {
                                        value: gainNode.gain.value,
                                        automation
                                    }
                                });
                            }
                        }
                        ctx._postMessage({
                            action: 'playBufferSource',
                            id: this._instanceId,
                            bufferData: this._getBufferData(),
                            loop: this.loop,
                            loopStart: this.loopStart,
                            loopEnd: this.loopEnd,
                            playbackRate: this.playbackRate.value,
                            offset: offset || 0,
                            gainChain: gainChain
                        });
                        console.log('AudioContext2: bufferSource.start at', startAt.toFixed(3), '-> playBufferSource', 'loop:', this.loop, 'playbackRate:', this.playbackRate.value, 'offset:', offset || 0);
                    },
                    stop: function(when) {
                        if (this._stopped)
                            return;
                        const now = ctx.currentTime;
                        this._stopAt = (typeof when === 'number' ? when : now);
                        const shouldDelay = typeof when === 'number' && when > now;
                        if (!this._connectedToDestination) {
                            console.warn('AudioContext2: bufferSource not connected to destination, ignore stop');
                            return;
                        }
                        if (shouldDelay) {
                            const delayMs = Math.max(0, (when - now) * 1000);
                            ctx._sendDelayedMessage({
                                action: 'stopBufferSource',
                                id: this._instanceId
                            }, delayMs);
                            setTimeout( () => {
                                    this._stopped = true;
                                }
                                , delayMs);
                            console.log('AudioContext2: 延迟停止 at', when.toFixed(3), '(delay:', delayMs + 'ms)');
                        } else {
                            ctx._postMessage({
                                action: 'stopBufferSource',
                                id: this._instanceId
                            });
                            this._stopped = true;
                            console.log('AudioContext2: 立即停止 at', now.toFixed(3));
                        }
                    },
                    _getBufferData: function() {
                        if (!this.buffer)
                            return null;
                        const bufferData = {
                            sampleRate: this.buffer.sampleRate,
                            length: this.buffer.length,
                            numberOfChannels: this.buffer.numberOfChannels,
                            channels: []
                        };
                        for (let channel = 0; channel < this.buffer.numberOfChannels; channel++) {
                            const channelData = this.buffer.getChannelData(channel);
                            bufferData.channels.push(Array.from(channelData));
                        }
                        return bufferData;
                    }
                };
                return bufferSource;
            }
            createBuffer(numberOfChannels, length, sampleRate) {
                const buffer = {
                    sampleRate: sampleRate,
                    length: length,
                    numberOfChannels: numberOfChannels,
                    duration: length / sampleRate,
                    _channelData: [],
                    getChannelData: function(channel) {
                        if (!this._channelData[channel]) {
                            this._channelData[channel] = new Float32Array(this.length);
                        }
                        return this._channelData[channel];
                    },
                    copyFromChannel: function(destination, channelNumber, startInChannel) {
                        const source = this.getChannelData(channelNumber);
                        const start = startInChannel || 0;
                        for (let i = 0; i < destination.length && i + start < source.length; i++) {
                            destination[i] = source[i + start];
                        }
                    },
                    copyToChannel: function(source, channelNumber, startInChannel) {
                        const destination = this.getChannelData(channelNumber);
                        const start = startInChannel || 0;
                        for (let i = 0; i < source.length && i + start < destination.length; i++) {
                            destination[i + start] = source[i];
                        }
                    }
                };
                return buffer;
            }
        }
        window.AudioContext2 = AudioContext2;
        console.log('AudioContext2 已加载：支持完整的gainchain信息传递和AudioBufferSourceNode');
    }
)();
function callLLM(message, system_prompt=null, timeout=60000) {
    return new Promise( (resolve, reject) => {
            const callId = 'candyJar_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
            if (!window._candyJarCallbacks) {
                window._candyJarCallbacks = {};
            }
            window._candyJarCallbacks[callId] = {
                resolve: (result) => {
                    clearTimeout(timeoutId);
                    delete window._candyJarCallbacks[callId];
                    resolve(result);
                }
                ,
                reject: (error) => {
                    clearTimeout(timeoutId);
                    delete window._candyJarCallbacks[callId];
                    reject(error);
                }
            };
            const timeoutId = setTimeout( () => {
                    if (window._candyJarCallbacks[callId]) {
                        console.warn('LLM 调用超时:', callId);
                        delete window._candyJarCallbacks[callId];
                        reject(new Error('LLM call timeout'));
                    }
                }
                , timeout);
            const extInfo = {
                callId: callId,
                click_html_type: 5
            };
            if (system_prompt) {
                extInfo.system_prompt = system_prompt;
            }
            const params = {
                scene: "API_INVOKE",
                query: message,
                appId: "202508200201904661365",
                extInfo: extInfo
            };
            console.log('发送LLM请求:', params);
            window.CandyJar.call('HtmlContent', 'requestInnerHtml', params, function(response) {
                console.log('LLM请求发送成功:', response);
            }, function(error) {
                console.error('LLM请求发送失败:', error);
                if (window._candyJarCallbacks[callId]) {
                    window._candyJarCallbacks[callId].reject(new Error('请求发送失败: ' + error));
                }
            }, timeout);
        }
    );
}
window.callLLM = callLLM;
if (!window._callLLMEventListenersInitialized) {
    window._callLLMEventListenersInitialized = true;
    window.document.addEventListener("CandyJar.HtmlContent.OnHtmlFragmentUpdate", function(event) {
        if (!event.htmlFragment || !event.htmlFragment.extInfo || event.htmlFragment.extInfo.click_html_type !== 5) {
            console.log('丢弃非llm api请求的OnHtmlFragmentUpdate事件');
            return;
        }
        if (!event.bubbles) {
            console.log('收到 bubbles=false 的事件，已丢弃');
            return;
        }
        console.log('子iframe接收到 CandyJar.HtmlContent.OnHtmlFragmentUpdate 事件，事件数据:', JSON.stringify(event, null, 2));
        if (event.htmlFragment && event.htmlFragment.success) {
            const callId = event.htmlFragment.extInfo && event.htmlFragment.extInfo.callId;
            const content = event.htmlFragment.content;
            const hasNext = event.htmlFragment.hasNext;
            if (callId) {
                console.log('处理流式响应 - callId:', callId, 'hasNext:', hasNext);
                if (!window._streamingResponses) {
                    window._streamingResponses = new Map();
                }
                if (!window._streamingResponses.has(callId)) {
                    window._streamingResponses.set(callId, '');
                }
                const currentContent = window._streamingResponses.get(callId);
                if (content) {
                    window._streamingResponses.set(callId, currentContent + content);
                }
                console.log('累积内容长度:', window._streamingResponses.get(callId).length);
                if (hasNext === false) {
                    console.log('流式响应结束 - callId:', callId);
                    const fullContent = window._streamingResponses.get(callId);
                    window._streamingResponses.delete(callId);
                    if (window._candyJarCallbacks && window._candyJarCallbacks[callId]) {
                        window._candyJarCallbacks[callId].resolve({
                            content: fullContent,
                            extInfo: event.htmlFragment.extInfo
                        });
                    }
                }
            }
        } else {
            const callId = event.htmlFragment && event.htmlFragment.extInfo && event.htmlFragment.extInfo.callId;
            if (callId) {
                console.log('LLM响应失败 - callId:', callId);
                if (window._streamingResponses) {
                    window._streamingResponses.delete(callId);
                }
                if (window._candyJarCallbacks && window._candyJarCallbacks[callId]) {
                    window._candyJarCallbacks[callId].reject(new Error('LLM响应失败: ' + (event.htmlFragment && event.htmlFragment.error || '未知错误')));
                }
            }
        }
    });
}
if (!window._webContainerResizeListenerInitialized) {
    window._webContainerResizeListenerInitialized = true;
    let originalState = null;
    window.document.addEventListener("CandyJar.WebContainer.Resize", function(event) {
        const action = event.action;
        if (!action || (action !== 'zoomin' && action !== 'zoomout'))
            return;
        try {
            if (!document.body)
                return;
            if (action === 'zoomin') {
                if (!originalState) {
                    originalState = {
                        overflow: document.body.style.overflow || '',
                        alignItems: document.body.style.alignItems || '',
                        containerPaddingTop: null
                    };
                    const bodyDivs = Array.from(document.body.children).filter(child => child.tagName === 'DIV');
                    if (bodyDivs.length === 1 && bodyDivs[0].id === 'container') {
                        originalState.containerPaddingTop = window.getComputedStyle(bodyDivs[0]).paddingTop;
                    }
                }
                if (document.body.style.alignItems === 'flex-start' || !document.body.style.alignItems) {
                    document.body.style.setProperty('align-items', 'center', 'important');
                }
                const bodyDivs = Array.from(document.body.children).filter(child => child.tagName === 'DIV');
                if (bodyDivs.length === 1 && bodyDivs[0].id === 'container') {
                    const onlyDiv = bodyDivs[0];
                    const rem = parseFloat(window.getComputedStyle(document.documentElement).fontSize);
                    const currentPaddingTop = parseFloat(window.getComputedStyle(onlyDiv).paddingTop) || 0;
                    const additionalPaddingRem = 88 / rem;
                    const newPaddingTopRem = (currentPaddingTop / rem) + additionalPaddingRem;
                    onlyDiv.style.setProperty('padding-top', `${newPaddingTopRem}rem`, 'important');
                }
            } else if (action === 'zoomout' && originalState) {
                if (originalState.alignItems === 'flex-start' || originalState.alignItems === '') {
                    document.body.style.setProperty('align-items', 'flex-start', 'important');
                } else if (originalState.alignItems) {
                    document.body.style.setProperty('align-items', originalState.alignItems, 'important');
                }
                const bodyDivs = Array.from(document.body.children).filter(child => child.tagName === 'DIV');
                if (bodyDivs.length === 1 && bodyDivs[0].id === 'container' && originalState.containerPaddingTop !== null) {
                    bodyDivs[0].style.setProperty('padding-top', originalState.containerPaddingTop, 'important');
                }
                originalState = null;
            }
        } catch (error) {
            console.error('样式调整操作失败:', error);
        }
    });
}
(function() {
        let lastHeight = 0;
        let isDebouncing = false;
        const urlParams = new URLSearchParams(window.location.search);
        const ua = navigator.userAgent.toLowerCase();
        const isIOS = /iphone|ipad|ipod/.test(ua);
        const isAndroid = /android/.test(ua);
        const isInIframe = window.self !== window.top;
        let OS_TYPE = 'pc';
        if (isIOS) {
            OS_TYPE = 'ios';
        } else if (isAndroid) {
            OS_TYPE = 'android';
        }
        const isFullScreen = urlParams.get('fullscreen') === 'true';
        const is_in_fr_raw = urlParams.get('is_in_fr');
        const is_in_fr = is_in_fr_raw === 'true';
        function fixBodyStyleForCompatibility() {
            if (document.body && isInIframe) {
                document.body.style.setProperty('margin', '0', 'important');
                document.body.style.setProperty('padding', '0', 'important');
                document.body.style.setProperty('display', 'flex', 'important');
                document.body.style.setProperty('align-items', 'flex-start', 'important');
                document.body.style.setProperty('justify-content', 'center', 'important');
            }
            if (document.body) {
                const contentTags = ['DIV', 'SECTION', 'ARTICLE', 'MAIN', 'ASIDE', 'NAV', 'HEADER', 'FOOTER', 'P', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'UL', 'OL', 'TABLE', 'FORM', 'CANVAS', 'SVG'];
                const bodyChildren = Array.from(document.body.children).filter(function(child) {
                    return child.nodeType === Node.ELEMENT_NODE && contentTags.includes(child.tagName);
                });
                if (bodyChildren.length === 1) {
                    bodyChildren[0].style.setProperty('margin', '0', 'important');
                    bodyChildren[0].classList.add("w-full");
                }
                if (isIOS) {
                    document.body.style.setProperty('-webkit-touch-callout', 'none', 'important');
                    document.body.style.setProperty('-webkit-user-select', 'none', 'important');
                }
            }
        }
        function preventContextMenu() {
            if (isAndroid && is_in_fr) {
                document.addEventListener('contextmenu', function(e) {
                    e.preventDefault();
                }, true);
            }
        }
        function sendHeightToParent() {
            if (!isInIframe)
                return;
            const height = Math.max(document.body.offsetHeight);
            if (height !== lastHeight) {
                window.parent.postMessage({
                    type: 'miniapp_resize',
                    height: height,
                    timestamp: Date.now()
                }, '*');
                console.log("发送高度信息:", height, "(上次:", lastHeight, ")");
                lastHeight = height;
            }
        }
        function getBackgroundColor() {
            const val = getComputedStyle(document.body).backgroundColor.trim();
            console.log("原始 backgroundColor 值:", val);
            if (val === 'transparent' || val === 'rgba(0, 0, 0, 0)') {
                console.log("body背景色透明，尝试获取第一个可视div的背景色");
                const divs = document.querySelectorAll('div');
                for (let div of divs) {
                    const divStyle = getComputedStyle(div);
                    const divBgColor = divStyle.backgroundColor.trim();
                    if (divStyle.display !== 'none' && divStyle.visibility !== 'hidden' && divBgColor !== 'transparent' && divBgColor !== 'rgba(0, 0, 0, 0)') {
                        console.log("找到可视div的背景色:", divBgColor);
                        const nums = divBgColor.match(/[\d.]+/g);
                        if (nums && nums.length >= 3) {
                            let[r,g,b] = nums.slice(0, 3).map(Number);
                            [r,g,b] = [r, g, b].map(n => Math.max(0, Math.min(255, Math.round(n))));
                            const toHex = n => n.toString(16).padStart(2, '0');
                            const result = `#${toHex(r)}${toHex(g)}${toHex(b)}`;
                            console.log("使用div背景色，转换为 hex:", result);
                            return result;
                        }
                    }
                }
                console.log("未找到可视div的背景色，返回默认白色");
                return '#ffffff';
            }
            const nums = val.match(/[\d.]+/g);
            console.log("提取的数字数组:", nums);
            if (!nums || nums.length < 3)
                return null;
            let[r,g,b] = nums.slice(0, 3).map(Number);
            console.log("解析的 RGB 值:", r, g, b);
            [r,g,b] = [r, g, b].map(n => Math.max(0, Math.min(255, Math.round(n))));
            const toHex = n => n.toString(16).padStart(2, '0');
            const result = `#${toHex(r)}${toHex(g)}${toHex(b)}`;
            console.log("最终转换的 hex 值:", result);
            return result;
        }
        function sendBackgroundColorToParent() {
            const colorToSend = getBackgroundColor();
            console.log("发送背景色:", colorToSend);
            window.parent.postMessage({
                type: 'miniapp_bgcolor',
                bgcolor: colorToSend
            }, '*');
        }
        function waitForStylesAndSendBgColor(maxRetries=5, delayMs=100) {
            let retryCount = 0;
            let lastColor = null;
            function checkAndSend() {
                const currentColor = getBackgroundColor();
                if (currentColor === lastColor || retryCount >= maxRetries) {
                    console.log("样式已稳定，发送背景色:", currentColor, "重试次数:", retryCount);
                    sendBackgroundColorToParent();
                    return;
                }
                lastColor = currentColor;
                retryCount++;
                setTimeout(checkAndSend, delayMs);
            }
            setTimeout(checkAndSend, delayMs);
        }
        function debouncedSendHeight() {
            if (isDebouncing)
                return;
            isDebouncing = true;
            setTimeout( () => {
                    sendHeightToParent();
                    isDebouncing = false;
                }
                , 150);
        }
        window.addEventListener('message', function(event) {
            if (event.data.type === 'checkHeight') {
                sendHeightToParent();
            }
        });
        function notifyFullscreenRenderEnd() {
            if (!isFullScreen) {
                console.log('[MiniApp] 当前不是全屏模式，跳过渲染完成通知');
                return;
            }
            console.log('[MiniApp] 准备通知父窗口全屏渲染完成');
            requestAnimationFrame( () => {
                    console.log('[MiniApp] 第一帧渲染完成');
                    requestAnimationFrame( () => {
                            console.log('[MiniApp] 第二帧渲染完成，发送 flashapp_fullscreen_render_end 消息');
                            window.parent.postMessage({
                                type: 'flashapp_fullscreen_render_end'
                            }, '*');
                        }
                    );
                }
            );
        }
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', function() {
                fixBodyStyleForCompatibility();
                sendHeightToParent();
                preventContextMenu();
                waitForStylesAndSendBgColor(5, 1000);
                notifyFullscreenRenderEnd();
            });
        } else {
            fixBodyStyleForCompatibility();
            sendHeightToParent();
            preventContextMenu();
            waitForStylesAndSendBgColor(5, 1000);
            notifyFullscreenRenderEnd();
        }
        const mutationObserver = new MutationObserver(function(mutations) {
                let shouldUpdate = false;
                mutations.forEach(function(mutation) {
                    if (mutation.type === 'childList' && (mutation.addedNodes.length > 0 || mutation.removedNodes.length > 0)) {
                        shouldUpdate = true;
                    } else if (mutation.type === 'attributes' && mutation.attributeName === 'style') {
                        shouldUpdate = true;
                    }
                });
                if (shouldUpdate) {
                    debouncedSendHeight();
                }
            }
        );
        mutationObserver.observe(document.body, {
            childList: true,
            subtree: true,
            attributes: true,
            attributeFilter: ['style']
        });
        window.addEventListener('error', function(e) {
            parent.postMessage({
                type: 'miniapp_error',
                error: {
                    message: e.message,
                    filename: e.filename,
                    lineno: e.lineno,
                    colno: e.colno
                }
            }, '*');
        });
        window.addEventListener('unhandledrejection', function(e) {
            parent.postMessage({
                type: 'miniapp_error',
                error: {
                    message: 'Unhandled Promise Rejection: ' + e.reason
                }
            }, '*');
        });
        if (OS_TYPE !== 'pc' && isInIframe && !isFullScreen) {
            (function() {
                    function isIOSDevice() {
                        const ua = navigator.userAgent || navigator.vendor || window.opera;
                        return /iPad|iPhone|iPod/.test(ua) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
                    }
                    let nestedScrollState = false;
                    let touchStartY = 0;
                    let lastTouchY = 0;
                    let scrollThreshold = 5;
                    let cooldownMs = 150;
                    let lastSwitchTime = 0;
                    let scrollContainer = null;
                    let lastScrollDirection = null;
                    function getScrollContainer() {
                        if (!scrollContainer) {
                            scrollContainer = document.scrollingElement || document.documentElement;
                        }
                        return scrollContainer;
                    }
                    function disableIOSBounce() {
                        if (isIOSDevice()) {
                            const container = getScrollContainer();
                            container.style.setProperty('overscroll-behavior', 'none', 'important');
                            container.style.setProperty('-webkit-overflow-scrolling', 'auto', 'important');
                            if (document.body) {
                                document.body.style.setProperty('overscroll-behavior', 'none', 'important');
                                document.body.style.setProperty('-webkit-overflow-scrolling', 'auto', 'important');
                            }
                            console.log('[嵌套滚动] iOS 设备检测，已禁用回弹效果');
                        }
                    }
                    disableIOSBounce();
                    function canScroll() {
                        const container = getScrollContainer();
                        return container.scrollHeight > container.clientHeight;
                    }
                    function isAtTop() {
                        if (!canScroll())
                            return false;
                        const container = getScrollContainer();
                        return container.scrollTop <= 2;
                    }
                    function isAtBottom() {
                        if (!canScroll())
                            return false;
                        const container = getScrollContainer();
                        const {scrollTop, scrollHeight, clientHeight} = container;
                        return scrollTop + clientHeight >= scrollHeight - 2;
                    }
                    function sendNestedScrollMessage(params, reason) {
                        nestedScrollState = params.enabled;
                        window.parent.postMessage({
                            type: 'nested_scroll',
                            params: params,
                            reason: reason
                        }, '*');
                        console.log(`[嵌套滚动] ${params.enabled ? '启用' : '禁用'} - 原因: ${reason}`, {
                            nestedScrollState: nestedScrollState,
                            params: params,
                            reason: reason,
                            timestamp: Date.now()
                        });
                    }
                    let isTracking = false;
                    document.addEventListener('touchstart', function(e) {
                        if (!isInIframe)
                            return;
                        touchStartY = e.touches[0].clientY;
                        lastTouchY = touchStartY;
                        isTracking = true;
                        console.log('[嵌套滚动] touchstart', {
                            touchStartY: touchStartY,
                            nestedScrollState: nestedScrollState
                        });
                        if (!nestedScrollState && canScroll()) {
                            console.log('[嵌套滚动] touchstart - 提前切换到 enable，确保能接收 touchmove');
                            sendNestedScrollMessage({
                                enabled: true
                            }, 'touchstart_preemptive');
                        }
                    }, {
                        passive: true
                    });
                    document.addEventListener('touchmove', function(e) {
                        console.log('[嵌套滚动] touchmove 事件触发', {
                            isTracking: isTracking,
                            isInIframe: isInIframe,
                            hasTouches: !!e.touches && e.touches.length > 0
                        });
                        if (!isTracking || !isInIframe) {
                            console.log('[嵌套滚动] touchmove - 条件不满足，返回', {
                                isTracking: isTracking,
                                isInIframe: isInIframe
                            });
                            return;
                        }
                        const currentY = e.touches[0].clientY;
                        const totalDeltaY = currentY - touchStartY;
                        if (Math.abs(totalDeltaY) < scrollThreshold) {
                            console.log('[嵌套滚动] touchmove - 滑动距离未达阈值', {
                                totalDeltaY: totalDeltaY.toFixed(2),
                                threshold: scrollThreshold
                            });
                            lastTouchY = currentY;
                            return;
                        }
                        const isScrollingDown = totalDeltaY > 0;
                        const isScrollingUp = totalDeltaY < 0;
                        if (isScrollingDown) {
                            lastScrollDirection = 'down';
                        } else if (isScrollingUp) {
                            lastScrollDirection = 'up';
                        }
                        if (!canScroll()) {
                            console.log('[嵌套滚动] touchmove - 内容不可滚动，不处理');
                            return;
                        }
                        const atTop = isAtTop();
                        const atBottom = isAtBottom();
                        console.log('[嵌套滚动] touchmove', {
                            totalDeltaY: totalDeltaY.toFixed(2),
                            isScrollingDown: isScrollingDown,
                            isScrollingUp: isScrollingUp,
                            atTop: atTop,
                            atBottom: atBottom,
                            nestedScrollState: nestedScrollState,
                            lastScrollDirection: lastScrollDirection
                        });
                        if (isScrollingDown) {
                            sendNestedScrollMessage({
                                enabled: true,
                                direction: 'down'
                            }, 'downStart');
                        } else if (isScrollingUp) {
                            sendNestedScrollMessage({
                                enabled: true,
                                direction: 'up'
                            }, 'upStart');
                        }
                        lastTouchY = currentY;
                    }, {
                        passive: true
                    });
                    function handleTouchEnd(reason) {
                        console.log(`[嵌套滚动] ${reason}`, {
                            nestedScrollState: nestedScrollState
                        });
                        nestedScrollState = false;
                        const atTop = isAtTop();
                        const atBottom = isAtBottom();
                        if (atBottom) {
                            console.log(`[嵌套滚动] ${reason} - 到达底部，立即切换回 disable`);
                            sendNestedScrollMessage({
                                enabled: false,
                                direction: lastScrollDirection || 'down'
                            }, 'reachBottom');
                        } else if (atTop) {
                            console.log(`[嵌套滚动] ${reason} - 到达顶部，立即切换回 disable`);
                            sendNestedScrollMessage({
                                enabled: false,
                                direction: lastScrollDirection || 'up'
                            }, 'reachTop');
                        } else {
                            console.log(`[嵌套滚动] ${reason} - 切换回 disable`);
                            sendNestedScrollMessage({
                                enabled: false,
                                direction: lastScrollDirection || 'down'
                            }, reason);
                        }
                        isTracking = false;
                    }
                    document.addEventListener('touchend', function() {
                        handleTouchEnd('touchend');
                    }, {
                        passive: true
                    });
                    document.addEventListener('touchcancel', function() {
                        handleTouchEnd('touchcancel');
                    }, {
                        passive: true
                    });
                    function checkScrollBoundaries() {
                        if (!nestedScrollState)
                            return;
                        const atTop = isAtTop();
                        const atBottom = isAtBottom();
                        if (atTop) {
                            console.log('[嵌套滚动] checkScrollBoundaries - 到达顶部');
                            sendNestedScrollMessage({
                                enabled: false,
                                direction: lastScrollDirection || 'up'
                            }, 'reachTop');
                        } else if (atBottom) {
                            console.log('[嵌套滚动] checkScrollBoundaries - 到达底部');
                            sendNestedScrollMessage({
                                enabled: false,
                                direction: lastScrollDirection || 'down'
                            }, 'reachBottom');
                        }
                    }
                    getScrollContainer().addEventListener('scroll', function() {
                        checkScrollBoundaries();
                    }, {
                        passive: true
                    });
                }
            )();
        }
    }
)();
