#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

AUTHOR = u'Andr\xe9 F. T. Martins'
SITENAME = u'Andr\xe9 F. T. Martins'
SITEURL = ''

PATH = 'content'

STATIC_PATHS = ['images', 'docs']

TIMEZONE = 'Europe/Lisbon'

DEFAULT_LANG = u'en'

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

MENUITEMS = (
    ('Home', '/index.html'),
    ('Jobs', '/pages/jobs.html'),
    ('Publications', '/pages/publications.html'),
    ('Software', '/pages/software.html'),
    ('Courses', '/pages/courses.html'),
    ('SARDINE Lab', '/pages/sardine.html')
)

# Blogroll
LINKS = None #() #(('Pelican', 'http://getpelican.com/'),
         #('Python.org', 'http://python.org/'),
         #('Jinja2', 'http://jinja.pocoo.org/'),
         #('You can modify those links in your config file', '#'),)

# Social widget
SOCIAL = None #(('github', 'http://github.com/andre-martins'),)

DEFAULT_PAGINATION = False

# Uncomment following line if you want document-relative URLs when developing
RELATIVE_URLS = True

DISPLAY_CATEGORIES_ON_MENU = False
DISPLAY_PAGES_ON_MENU = False

MARKDOWN = {
    'extension_configs': {
        'markdown.extensions.tables': {},
    }
}
