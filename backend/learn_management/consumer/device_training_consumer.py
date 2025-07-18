from utils.rabbitmq_client import RabbitMQClient
from loguru import logger
import json
import traceback
from django.db import transaction

class DeviceTrainingMqConsumer(RabbitMQClient):
    def callback(self, ch, method, properties, body):

        try:
            message = json.loads(body.decode())
            logger.info(f"DeviceTrainingConsumer Received: {message}")
            device_id = message.get("device_id")
            region_id = message.get("region_id")
            device_context = message.get("device_context")
            training_context = message.get("training_context")
            task_id = message.get("task_id")
            model_id = message.get("model_id")
            model_version_id = message.get("model_version_id")
            training_status = message.get("training_status")

            if not all([device_id, region_id, device_context, training_context, task_id, model_id, model_version_id, training_status]):
                logger.error("DeviceTrainingConsumer Error: missing params")
                return
            # 调用注册接口
            self.device_training(device_id, region_id, device_context, training_context, task_id, model_id, model_version_id, training_status)
        except Exception as e:
            traceback.print_exc()
            logger.error(f"DeviceTrainingConsumer Error: {e}")

    def device_training(self, device_id, region_id, device_context, training_context, task_id, model_id, model_version_id,
                        training_status):
        from learn_management.models import FederatedTask, ModelInfo, ModelVersion, RegionNode, EdgeNode, TrainingRecord
        try:
            logger.info(f"Processing training task for device: {device_id}, status: {training_status}")

            # 获取相关模型对象
            task = FederatedTask.objects.get(id=task_id)
            model_info = ModelInfo.objects.get(id=model_id)
            model_version = ModelVersion.objects.get(id=model_version_id)
            region_node = RegionNode.objects.get(id=region_id)

            # 获取或创建 EdgeNode
            edge_node, created = EdgeNode.objects.get_or_create(
                device_id=device_id,
                region_node=region_node,
                defaults={
                    'ip_address': device_context.get('ip'),
                    'device_context': device_context,
                    'status': 'online'
                }
            )

            if not created:
                edge_node.device_context = device_context
                edge_node.status = 'online'
                edge_node.save()

            # 根据状态处理训练记录
            if training_status == 'running':
                # 创建训练记录（开始）
                with transaction.atomic():
                    training_record = TrainingRecord.objects.create(
                        edge_node=edge_node,
                        federated_task=task,
                        model_info=model_info,
                        model_version=model_version,
                        region_node=region_node,
                        device_context=device_context,
                        training_context=training_context,
                        status='running',
                    )
                logger.info(f"Training record started: {training_record.id}")

            elif training_status in ['completed', 'failed', 'paused', 'canceled']:
                # 更新训练记录（完成或失败）
                try:
                    training_record = TrainingRecord.objects.filter(
                        edge_node=edge_node,
                        federated_task=task,
                        model_info=model_info,
                        model_version=model_version,
                        status='running'
                    ).first()

                    if training_record:
                        training_record.status = training_status
                        training_record.save()

                    logger.info(f"Training record updated: {training_record.id}, duration: {training_record.duration}s")
                except TrainingRecord.DoesNotExist:
                    logger.warning("No running training record found to update.")
        except Exception as e:
            logger.error(f"Error processing training status: {e}")

