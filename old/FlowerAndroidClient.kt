// FlowerAndroidClient.kt - Android客户端实现
package com.example.flowerfl

import android.content.Context
import android.util.Log
import kotlinx.coroutines.*
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.io.*
import java.net.*
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Executors
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken

class FlowerAndroidClient(
    private val context: Context,
    private val clientId: String,
    private val serverAddress: String = "192.168.1.100:8080"
) {
    companion object {
        private const val TAG = "FlowerAndroidClient"
        private const val MODEL_PATH = "simple_model.tflite"
        private const val INPUT_SIZE = 784 // 28x28 for MNIST
        private const val OUTPUT_SIZE = 10
        private const val BATCH_SIZE = 32
    }

    private var interpreter: Interpreter? = null
    private var isTraining = false
    private var localDataset: MutableList<TrainingSample> = mutableListOf()
    private val gson = Gson()

    // 训练样本数据类
    data class TrainingSample(
        val input: FloatArray,
        val label: Int
    )

    // 通信消息格式
    data class FLMessage(
        val type: String,
        val clientId: String,
        val round: Int? = null,
        val parameters: List<List<Float>>? = null,
        val metrics: Map<String, Float>? = null,
        val numSamples: Int? = null
    )

    // 设备信息
    data class DeviceInfo(
        val clientId: String,
        val deviceType: String = "android",
        val model: String = android.os.Build.MODEL,
        val manufacturer: String = android.os.Build.MANUFACTURER,
        val androidVersion: String = android.os.Build.VERSION.RELEASE,
        val cpuCores: Int = Runtime.getRuntime().availableProcessors(),
        val memoryMB: Long = getAvailableMemoryMB()
    )

    init {
        initializeModel()
        generateLocalDataset()
    }

    private fun initializeModel() {
        try {
            // 从assets加载TensorFlow Lite模型
            val modelBuffer = FileUtil.loadMappedFile(context, MODEL_PATH)
            interpreter = Interpreter(modelBuffer)

            Log.i(TAG, "Model initialized successfully")
            Log.i(TAG, "Input tensor count: ${interpreter?.inputTensorCount}")
            Log.i(TAG, "Output tensor count: ${interpreter?.outputTensorCount}")

        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize model: ${e.message}")
            // 创建简单的虚拟模型用于演示
            createDummyModel()
        }
    }

    private fun createDummyModel() {
        // 为演示创建虚拟模型权重
        Log.w(TAG, "Using dummy model for demonstration")
    }

    private fun generateLocalDataset() {
        // 生成模拟的本地数据集
        val random = kotlin.random.Random(clientId.hashCode())

        repeat(1000) { // 生成1000个样本
            val input = FloatArray(INPUT_SIZE) { random.nextFloat() }
            val label = random.nextInt(OUTPUT_SIZE)
            localDataset.add(TrainingSample(input, label))
        }

        Log.i(TAG, "Generated ${localDataset.size} local training samples")
    }

    // 获取设备可用内存
    private fun getAvailableMemoryMB(): Long {
        val runtime = Runtime.getRuntime()
        return (runtime.maxMemory() - runtime.totalMemory() + runtime.freeMemory()) / (1024 * 1024)
    }

    // 获取模型参数
    private fun getModelParameters(): List<List<Float>> {
        // 这里应该从TensorFlow Lite模型中提取参数
        // 为演示目的，返回模拟参数
        return listOf(
            List(784) { kotlin.random.Random.nextFloat() }, // 输入层权重
            List(128) { kotlin.random.Random.nextFloat() }, // 隐藏层权重
            List(10) { kotlin.random.Random.nextFloat() }   // 输出层权重
        )
    }

    // 设置模型参数
    private fun setModelParameters(parameters: List<List<Float>>) {
        // 这里应该将参数设置到TensorFlow Lite模型中
        Log.i(TAG, "Setting model parameters with ${parameters.size} layers")
    }

    // 本地训练
    suspend fun performLocalTraining(
        globalParameters: List<List<Float>>,
        localEpochs: Int = 1
    ): Pair<List<List<Float>>, Map<String, Float>> = withContext(Dispatchers.Default) {

        Log.i(TAG, "Starting local training for $localEpochs epochs")
        isTraining = true

        try {
            // 设置全局参数
            setModelParameters(globalParameters)

            var totalLoss = 0f
            var totalAccuracy = 0f
            var processedSamples = 0

            // 简化的训练循环
            for (epoch in 0 until localEpochs) {
                val shuffledData = localDataset.shuffled()
                var epochLoss = 0f
                var epochAccuracy = 0f
                var batchCount = 0

                // 批量处理
                for (i in shuffledData.indices step BATCH_SIZE) {
                    val batchEnd = minOf(i + BATCH_SIZE, shuffledData.size)
                    val batch = shuffledData.subList(i, batchEnd)

                    // 模拟训练一个批次
                    val batchResult = trainBatch(batch)
                    epochLoss += batchResult.first
                    epochAccuracy += batchResult.second
                    batchCount++

                    processedSamples += batch.size

                    // 检查是否需要暂停（电池优化）
                    if (shouldPauseTraining()) {
                        Log.w(TAG, "Training paused due to battery optimization")
                        delay(1000)
                    }
                }

                epochLoss /= batchCount
                epochAccuracy /= batchCount
                totalLoss += epochLoss
                totalAccuracy += epochAccuracy

                Log.i(TAG, "Epoch ${epoch + 1}/$localEpochs - Loss: $epochLoss, Accuracy: $epochAccuracy")
            }

            val avgLoss = totalLoss / localEpochs
            val avgAccuracy = totalAccuracy / localEpochs

            val metrics = mapOf(
                "train_loss" to avgLoss,
                "train_accuracy" to avgAccuracy,
                "num_samples" to processedSamples.toFloat(),
                "local_epochs" to localEpochs.toFloat(),
                "device_type" to 1f // Android = 1
            )

            Log.i(TAG, "Local training completed. Loss: $avgLoss, Accuracy: $avgAccuracy")

            return@withContext Pair(getModelParameters(), metrics)

        } finally {
            isTraining = false
        }
    }

    // 训练单个批次
    private fun trainBatch(batch: List<TrainingSample>): Pair<Float, Float> {
        // 模拟批次训练
        val loss = kotlin.random.Random.nextFloat() * 0.5f + 0.1f
        val accuracy = kotlin.random.Random.nextFloat() * 0.3f + 0.7f
        return Pair(loss, accuracy)
    }

    // 检查是否需要暂停训练（电池优化）
    private fun shouldPauseTraining(): Boolean {
        // 检查电池电量、充电状态等
        return false // 简化实现
    }

    // 模型评估
    suspend fun evaluateModel(parameters: List<List<Float>>): Map<String, Float> = withContext(Dispatchers.Default) {

        Log.i(TAG, "Starting model evaluation")

        setModelParameters(parameters)

        // 使用部分数据进行评估
        val evalData = localDataset.take(200)
        var totalLoss = 0f
        var correct = 0

        for (sample in evalData) {
            // 模拟评估
            val prediction = kotlin.random.Random.nextInt(OUTPUT_SIZE)
            if (prediction == sample.label) correct++
            totalLoss += kotlin.random.Random.nextFloat() * 0.3f
        }

        val accuracy = correct.toFloat() / evalData.size
        val avgLoss = totalLoss / evalData.size

        Log.i(TAG, "Evaluation completed. Loss: $avgLoss, Accuracy: $accuracy")

        return@withContext mapOf(
            "eval_loss" to avgLoss,
            "eval_accuracy" to accuracy,
            "eval_samples" to evalData.size.toFloat()
        )
    }

    // 连接到联邦学习服务器
    suspend fun connectToServer() = withContext(Dispatchers.IO) {
        Log.i(TAG, "Connecting to FL server at $serverAddress")

        try {
            val socket = Socket()
            socket.connect(InetSocketAddress(serverAddress.split(":")[0],
                         serverAddress.split(":")[1].toInt()), 5000)

            val writer = PrintWriter(socket.getOutputStream(), true)
            val reader = BufferedReader(InputStreamReader(socket.getInputStream()))

            // 发送设备注册信息
            val deviceInfo = DeviceInfo(clientId)
            val registerMessage = FLMessage("register", clientId)
            writer.println(gson.toJson(registerMessage))

            // 处理服务器消息
            handleServerCommunication(reader, writer)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to connect to server: ${e.message}")
            throw e
        }
    }

    // 处理与服务器的通信
    private suspend fun handleServerCommunication(
        reader: BufferedReader,
        writer: PrintWriter
    ) = withContext(Dispatchers.IO) {

        while (true) {
            try {
                val messageJson = reader.readLine() ?: break
                val message = gson.fromJson(messageJson, FLMessage::class.java)

                when (message.type) {
                    "train" -> {
                        Log.i(TAG, "Received training request for round ${message.round}")

                        val parameters = message.parameters ?: emptyList()
                        val (updatedParams, metrics) = performLocalTraining(parameters)

                        val response = FLMessage(
                            type = "train_result",
                            clientId = clientId,
                            round = message.round,
                            parameters = updatedParams,
                            metrics = metrics,
                            numSamples = localDataset.size
                        )

                        writer.println(gson.toJson(response))
                    }

                    "evaluate" -> {
                        Log.i(TAG, "Received evaluation request")

                        val parameters = message.parameters ?: emptyList()
                        val evalMetrics = evaluateModel(parameters)

                        val response = FLMessage(
                            type = "eval_result",
                            clientId = clientId,
                            metrics = evalMetrics,
                            numSamples = 200
                        )

                        writer.println(gson.toJson(response))
                    }

                    "disconnect" -> {
                        Log.i(TAG, "Server requested disconnection")
                        break
                    }

                    else -> {
                        Log.w(TAG, "Unknown message type: ${message.type}")
                    }
                }

            } catch (e: Exception) {
                Log.e(TAG, "Error handling server communication: ${e.message}")
                break
            }
        }
    }

    // 启动联邦学习客户端
    fun startFederatedLearning() {
        CoroutineScope(Dispatchers.IO).launch {
            try {
                connectToServer()
            } catch (e: Exception) {
                Log.e(TAG, "Failed to start federated learning: ${e.message}")
            }
        }
    }

    // 停止客户端
    fun stopClient() {
        isTraining = false
        interpreter?.close()
        Log.i(TAG, "Flower Android client stopped")
    }

    // 获取客户端状态
    fun getClientStatus(): Map<String, Any> {
        return mapOf(
            "clientId" to clientId,
            "isTraining" to isTraining,
            "localDataSize" to localDataset.size,
            "deviceInfo" to DeviceInfo(clientId)
        )
    }
}