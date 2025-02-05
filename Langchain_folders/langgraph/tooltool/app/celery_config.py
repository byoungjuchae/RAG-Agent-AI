from celery import Celery


celery_app = Celery('task',
                    backend= 'https:localhost:6379')