from rest_framework import serializers
from WebAnalyzer.models import *


class ResultSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = ResultModel
        fields = ('rank','name','similarity' ,'image')


class ImageSerializer(serializers.HyperlinkedModelSerializer):
    result = ResultSerializer(many=True, read_only=True)

    class Meta:
        model = ImageModel
        fields = ('image','token','options','url','uploaded_date', 'updated_date', 'result')
        read_only_fields = ('token', 'uploaded_date', 'updated_date', 'result')
