from rest_framework import status
from rest_framework.pagination import PageNumberPagination
from rest_framework.response import Response


class CustomPagination(PageNumberPagination):
    code = status.HTTP_200_OK
    msg = "success"
    page_size_query_param = "page_size"
    page_size = 20

    def get_paginated_response(self, data):
        return Response(
            {
                "code": self.code,
                "msg": self.msg,
                "data": {
                    "list": data,
                    "page": self.page.number,
                    "total": self.page.paginator.count,
                },
            }
        )
