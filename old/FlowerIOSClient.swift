// FlowerIOSClient.swift - iOS客户端实现
import Foundation
import TensorFlowLite
import UIKit
import Network

class FlowerIOSClient: NSObject {

    // MARK: - Properties
    private let clientId: String
    private let serverAddress: String
    private var interpreter: Interpreter?
    private var isTraining = false
    private var localDataset: [TrainingSample] = []
    private var connection: NWConnection?

    // 常量
    private static let inputSize = 784 // 28x28 for MNIST
    private static let outputSize = 10
    private static let batchSize = 32
    private static let modelPath = "simple_model.tflite"

    // MARK: - Data Structures
    struct TrainingSample {
        let input: [Float]
        let label: Int
    }

    struct FLMessage: Codable {
        let type: String
        let clientId: String
        let round: Int?
        let parameters: [[Float]]?
        let metrics: [String: Float]?
        let numSamples: Int?

        init(type: String, clientId: String, round: Int? = nil,
             parameters: [[Float]]? = nil, metrics: [String: Float]? = nil,
             numSamples: Int? = nil) {
            self.type = type
            self.clientId = clientId
            self.round = round
            self.parameters = parameters
            self.metrics = metrics
            self.numSamples = numSamples
        }
    }

    struct DeviceInfo: Codable {
        let clientId: String
        let deviceType: String = "ios"
        let model: String
        let systemName: String
        let systemVersion: String
        let processorCount: Int
        let memoryMB: UInt64

        init(clientId: String) {
            self.clientId = clientId
            self.model = UIDevice.current.model
            self.systemName = UIDevice.current.systemName
            self.systemVersion = UIDevice.current.systemVersion
            self.processorCount = ProcessInfo.processInfo.processorCount
            self.memoryMB = ProcessInfo.processInfo.physicalMemory / (1024 * 1024)
        }
    }

    // MARK: - Initialization
    init(clientId: String, serverAddress: String = "192.168.1.100:8080") {
        self.clientId = clientId
        self.serverAddress = serverAddress
        super.init()

        initializeModel()
        generateLocalDataset()
    }

    // MARK: - Model Management
    private func initializeModel() {
        guard let modelPath = Bundle.main.path(forResource: "simple_model", ofType: "tflite") else {
            print("❌ Failed to load model file")
            createDummyModel()
            return
        }

        do {
            interpreter = try Interpreter(modelPath: modelPath)
            try interpreter?.allocateTensors()

            print("✅ Model initialized successfully")
            print("📊 Input tensor count: \(interpreter?.inputTensorCount ?? 0)")
            print("📊 Output tensor count: \(interpreter?.outputTensorCount ?? 0)")

        } catch {
            print("❌ Failed to initialize TensorFlow Lite interpreter: \(error)")
            createDummyModel()
        }
    }

    private func createDummyModel() {
        print("⚠️ Using dummy model for demonstration")
        // 创建虚拟模型用于演示
    }

    private func generateLocalDataset() {
        // 生成模拟的本地数据集
        var rng = SystemRandomNumberGenerator()

        for _ in 0..<1000 {
            let input = (0..<FlowerIOSClient.inputSize).map { _ in Float.random(in: 0...1, using: &rng) }
            let label = Int.random(in: 0..<FlowerIOSClient.outputSize, using: &rng)
            localDataset.append(TrainingSample(input: input, label: label))
        }

        print("📱 Generated \(localDataset.count) local training samples")
    }

    // MARK: - Model Operations
    private func getModelParameters() -> [[Float]] {
        // 在实际实现中，这里应该从TensorFlow Lite模型中提取参数
        // 为演示目的，返回模拟参数
        return [
            (0..<784).map { _ in Float.random(in: -1...1) }, // 输入层权重
            (0..<128).map { _ in Float.random(in: -1...1) }, // 隐藏层权重
            (0..<10).map { _ in Float.random(in: -1...1) }   // 输出层权重
        ]
    }

    private func setModelParameters(_ parameters: [[Float]]) {
        // 在实际实现中，这里应该将参数设置到TensorFlow Lite模型中
        print("🔧 Setting model parameters with \(parameters.count) layers")
    }

    // MARK: - Training
    func performLocalTraining(globalParameters: [[Float]], localEpochs: Int = 1) async -> ([[Float]], [String: Float]) {
        print("🚀 Starting local training for \(localEpochs) epochs")
        isTraining = true

        defer {
            isTraining = false
        }

        // 设置全局参数
        setModelParameters(globalParameters)

        var totalLoss: Float = 0
        var totalAccuracy: Float = 0
        var processedSamples = 0

        for epoch in 0..<localEpochs {
            let shuffledData = localDataset.shuffled()
            var epochLoss: Float = 0
            var epochAccuracy: Float = 0
            var batchCount = 0

            // 批量处理
            for i in stride(from: 0, to: shuffledData.count, by: FlowerIOSClient.batchSize) {
                let batchEnd = min(i + FlowerIOSClient.batchSize, shuffledData.count)
                let batch = Array(shuffledData[i..<batchEnd])

                // 模拟训练一个批次
                let batchResult = await trainBatch(batch)
                epochLoss += batchResult.0
                epochAccuracy += batchResult.1
                batchCount += 1

                processedSamples += batch.count

                // 检查是否需要暂停训练（电池优化）
                if shouldPauseTraining() {
                    print("⏸️ Training paused due to battery optimization")
                    try? await Task.sleep(nanoseconds: 1_000_000_000) // 1 second
                }
            }

            epochLoss /= Float(batchCount)
            epochAccuracy /= Float(batchCount)
            totalLoss += epochLoss
            totalAccuracy += epochAccuracy

            print("📈 Epoch \(epoch + 1)/\(localEpochs) - Loss: \(epochLoss), Accuracy: \(epochAccuracy)")
        }

        let avgLoss = totalLoss / Float(localEpochs)
        let avgAccuracy = totalAccuracy / Float(localEpochs)

        let metrics: [String: Float] = [
            "train_loss": avgLoss,
            "train_accuracy": avgAccuracy,
            "num_samples": Float(processedSamples),
            "local_epochs": Float(localEpochs),
            "device_type": 2.0 // iOS = 2
        ]

        print("✅ Local training completed. Loss: \(avgLoss), Accuracy: \(avgAccuracy)")

        return (getModelParameters(), metrics)
    }

    private func trainBatch(_ batch: [TrainingSample]) async -> (Float, Float) {
        // 模拟批次训练
        let loss = Float.random(in: 0.1...0.6)
        let accuracy = Float.random(in: 0.7...1.0)
        return (loss, accuracy)
    }

    private func shouldPauseTraining() -> Bool {
        // 检查电池电量、低功耗模式等
        let batteryLevel = UIDevice.current.batteryLevel
        let isLowPowerModeEnabled = ProcessInfo.processInfo.isLowPowerModeEnabled

        return batteryLevel < 0.2 || isLowPowerModeEnabled
    }

    // MARK: - Evaluation
    func evaluateModel(parameters: [[Float]]) async -> [String: Float] {
        print("📊 Starting model evaluation")

        setModelParameters(parameters)

        // 使用部分数据进行评估
        let evalData = Array(localDataset.prefix(200))
        var totalLoss: Float = 0
        var correct = 0

        for sample in evalData {
            // 模拟评估
            let prediction = Int.random(in: 0..<FlowerIOSClient.outputSize)
            if prediction == sample.label {
                correct += 1
            }
            totalLoss += Float.random(in: 0...0.3)
        }

        let accuracy = Float(correct) / Float(evalData.count)
        let avgLoss = totalLoss / Float(evalData.count)

        print("✅ Evaluation completed. Loss: \(avgLoss), Accuracy: \(accuracy)")

        return [
            "eval_loss": avgLoss,
            "eval_accuracy": accuracy,
            "eval_samples": Float(evalData.count)
        ]
    }

    // MARK: - Network Communication
    func connectToServer() async throws {
        print("🌐 Connecting to FL server at \(serverAddress)")

        let components = serverAddress.components(separatedBy: ":")
        guard components.count == 2,
              let host = components.first,
              let port = Int(components.last!) else {
            throw NSError(domain: "InvalidAddress", code: 1, userInfo: nil)
        }

        connection = NWConnection(host: NWEndpoint.Host(host),
                                 port: NWEndpoint.Port(integerLiteral: UInt16(port)),
                                 using: .tcp)

        connection?.stateUpdateHandler = { [weak self] state in
            switch state {
            case .ready:
                print("✅ Connected to server")
                Task {
                    await self?.handleServerCommunication()
                }
            case .failed(let error):
                print("❌ Connection failed: \(error)")
            case .cancelled:
                print("🔌 Connection cancelled")
            default:
                break
            }
        }

        connection?.start(queue: .global())

        // 发送设备注册信息
        let deviceInfo = DeviceInfo(clientId: clientId)
        let registerMessage = FLMessage(type: "register", clientId: clientId)
        try await sendMessage(registerMessage)
    }

    private func sendMessage(_ message: FLMessage) async throws {
        let encoder = JSONEncoder()
        let data = try encoder.encode(message)
        let messageWithNewline = data + "\n".data(using: .utf8)!

        await withCheckedContinuation { continuation in
            connection?.send(content: messageWithNewline, completion: .contentProcessed { error in
                if let error = error {
                    print("❌ Failed to send message: \(error)")
                }
                continuation.resume()
            })
        }
    }

    private func receiveMessage() async -> FLMessage? {
        return await withCheckedContinuation { continuation in
            connection?.receive(minimumIncompleteLength: 1, maximumLength: 65536) { data, _, isComplete, error in
                if let error = error {
                    print("❌ Failed to receive message: \(error)")
                    continuation.resume(returning: nil)
                    return
                }

                guard let data = data,
                      let jsonString = String(data: data, encoding: .utf8) else {
                    continuation.resume(returning: nil)
                    return
                }

                let decoder = JSONDecoder()
                do {
                    let message = try decoder.decode(FLMessage.self, from: data)
                    continuation.resume(returning: message)
                } catch {
                    print("❌ Failed to decode message: \(error)")
                    continuation.resume(returning: nil)
                }
            }
        }
    }

    private func handleServerCommunication() async {
        while connection?.state == .ready {
            guard let message = await receiveMessage() else {
                break
            }

            switch message.type {
            case "train":
                print("📚 Received training request for round \(message.round ?? 0)")

                let parameters = message.parameters ?? []
                let (updatedParams, metrics) = await performLocalTraining(
                    globalParameters: parameters
                )

                let response = FLMessage(
                    type: "train_result",
                    clientId: clientId,
                    round: message.round,
                    parameters: updatedParams,
                    metrics: metrics,
                    numSamples: localDataset.count
                )

                try? await sendMessage(response)

            case "evaluate":
                print("📊 Received evaluation request")

                let parameters = message.parameters ?? []
                let evalMetrics = await evaluateModel(parameters: parameters)

                let response = FLMessage(
                    type: "eval_result",
                    clientId: clientId,
                    metrics: evalMetrics,
                    numSamples: 200
                )

                try? await sendMessage(response)

            case "disconnect":
                print("👋 Server requested disconnection")
                return

            default:
                print("⚠️ Unknown message type: \(message.type)")
            }
        }
    }

    // MARK: - Public Interface
    func startFederatedLearning() {
        Task {
            do {
                try await connectToServer()
            } catch {
                print("❌ Failed to start federated learning: \(error)")
            }
        }
    }

    func stopClient() {
        isTraining = false
        connection?.cancel()
        print("🛑 Flower iOS client stopped")
    }

    func getClientStatus() -> [String: Any] {
        return [
            "clientId": clientId,
            "isTraining": isTraining,
            "localDataSize": localDataset.count,
            "deviceInfo": DeviceInfo(clientId: clientId)
        ]
    }
}