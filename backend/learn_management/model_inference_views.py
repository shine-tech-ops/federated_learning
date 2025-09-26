from rest_framework import generics, status
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from .models import ModelInferenceLog
from .serializers import ModelInferenceLogSerializer

class ModelInferenceLogView(generics.ListCreateAPIView, generics.RetrieveUpdateDestroyAPIView):
    """
    模型推理日志接口
    """
    queryset = ModelInferenceLog.objects.all().order_by('-created_at')
    serializer_class = ModelInferenceLogSerializer
    permission_classes = [IsAuthenticated]
    filterset_fields = ['model_version', 'edge_node']
    search_fields = ['error_message']
    ordering_fields = ['created_at', 'updated_at']

    def get_queryset(self):
        queryset = super().get_queryset()
        # 添加过滤条件
        model_version = self.request.query_params.get('model_version', None)
        if model_version:
            queryset = queryset.filter(model_version_id=model_version)
        
        edge_node = self.request.query_params.get('edge_node', None)
        if edge_node:
            queryset = queryset.filter(edge_node_id=edge_node)
            
        return queryset

    def perform_create(self, serializer):
        serializer.save(created_by=self.request.user)

    def create(self, request, *args, **kwargs):
        """
        创建推理日志
        """
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)
        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)