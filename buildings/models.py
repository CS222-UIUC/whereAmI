from django.db import models
import random

# Create your models here.

class Comment(models.Model):
    user = models.CharField(max_length=200, null=True)
    body = models.TextField()
    created_on = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_on']

    def __str__(self):
        return 'Comment "{}" by anonymous'.format(self.body)