# -*- coding: utf-8 -*-
# Generated by Django 1.11 on 2018-10-09 07:23
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='extractorModel',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=32)),
                ('url', models.URLField()),
                ('content', models.TextField(blank=True)),
                ('status', models.BooleanField(default=True)),
            ],
        ),
    ]