from django.shortcuts import render

# Create your views here.

from extractorManager.models import *
from extractorManager.serializers import *
from rest_framework import viewsets


class extractorViewSet(viewsets.ReadOnlyModelViewSet):

    queryset = extractorModel.objects.all()
    serializer_class = extractorSerializer

    def get_queryset(self):
        queryset = self.queryset

        name = self.request.query_params.get('name', None)
        if name is not None:
            queryset = queryset.filter(name__contains=name)

        url = self.request.query_params.get('url', None)
        if url is not None:
            queryset = queryset.filter(url__contains=url)

        status = self.request.query_params.get('status', None)
        if status is not None:
            status = 1 if status == 'true' else 0
            queryset = queryset.filter(status__exact=status)

        return queryset

