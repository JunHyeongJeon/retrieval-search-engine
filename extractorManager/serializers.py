from django.forms import widgets
from rest_framework import serializers
from extractorManager.models import *


class extractorSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = extractorModel
        fields = ('name', 'url', 'content', 'status')

