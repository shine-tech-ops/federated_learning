from django.db import migrations, models
import django.db.models.deletion
from django.conf import settings


class Migration(migrations.Migration):

    dependencies = [
        ('learn_management', '0018_federatedtraininglog'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='ModelChatLog',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('input_text', models.TextField(verbose_name='模型输入')),
                ('output_text', models.TextField(blank=True, null=True, verbose_name='模型输出')),
                ('extra_data', models.JSONField(blank=True, default=dict, null=True, verbose_name='扩展数据')),
                ('created_at', models.DateTimeField(auto_now_add=True, verbose_name='创建时间')),
                ('updated_at', models.DateTimeField(auto_now=True, verbose_name='更新时间')),
                ('created_by', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='model_chat_logs_created_by', to=settings.AUTH_USER_MODEL, verbose_name='创建人')),
                ('model_info', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='chat_logs', to='learn_management.modelinfo', verbose_name='模型')),
                ('model_version', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='chat_logs', to='learn_management.modelversion', verbose_name='模型版本')),
            ],
            options={
                'verbose_name': '模型对话日志',
                'db_table': 'model_chat_log',
                'db_table_comment': '模型对话日志表',
                'ordering': ['-created_at', '-id'],
            },
        ),
        migrations.AddIndex(
            model_name='modelchatlog',
            index=models.Index(fields=['model_info', 'created_at'], name='model_info_created_idx'),
        ),
        migrations.AddIndex(
            model_name='modelchatlog',
            index=models.Index(fields=['model_version', 'created_at'], name='model_version_created_idx'),
        ),
    ]


