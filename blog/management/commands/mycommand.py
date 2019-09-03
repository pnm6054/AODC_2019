from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from blog.models import Post
from django.utils import timezone
from . import predict
import time

class Command(BaseCommand):
    def handle(self, *args, **kwargs):
        me = User.objects.get(username='admin')
        n = 0
        while True:
            predict.predict(n)
            print('media/test' + str(n) + '.png')
            Post.objects.create(author=me, title='Generation amount', text='Wind-Power generation amount for 72 hours.', published_date=timezone.now(), photo='test' + str(n) + '.png')
            time.sleep(3600 * 3)
            n += 1
