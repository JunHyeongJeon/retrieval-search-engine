from django.db import models

# Create your models here.

from rest_framework import exceptions
import requests

class extractorModel(models.Model):
    name=models.CharField(max_length=32)
    url=models.URLField()
    content=models.TextField(blank=True)
    status=models.BooleanField(default=True)

    def save(self, *args, **kwargs):
        super(extractorModel, self).save(*args, **kwargs)
        try:
            response = requests.get(self.url)
            self.status = response.ok
        except:
            raise exceptions.ValidationError('Cannot access URL. Check module URL.')

       # self.group.update_or_create(name=self.name, content=self.content)
        super(extractorModel, self).save()
