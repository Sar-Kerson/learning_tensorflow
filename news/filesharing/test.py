# from urllib import urlopen
# from HTMLParser import HTMLParser
# from
#
# class Scraper(HTMLParser):
#
#     in_h3 = False
#     in_link = False
#
#     def handle_starttag(self, tag, attrs):
#         attrs = dict(attrs)
#         if tag == 'h3':
#             self.in_h3 = True
#
#         if tag == 'a' and 'href' in attrs:
#             self.in_link = True
#             self.chunks = []
#             self.url = attrs['href']
#
#     def handle_data(self, data):
#         if self.in_link:
#             self.chunks.append(data)
#
#     def handle_endtag(self, tag):
#         if tag == 'h3':
#             self.in_h3 = False
#         if tag == 'a':
#             if self.in_h3 and self.in_link:

from urllib import urlopen
from bs4 import BeautifulSoup

text = urlopen('https://www.python.org/jobs').read()
soup = BeautifulSoup(text, 'html5lib')

jobs = set()
for header in soup('h3'):
    links = header('a', 'reference')
    if not links: continue
    link = links[0]
    jobs.add('%s (%s)' % (link.string, link['href']))

print('\n'.join(sorted(jobs, key=lambda s: s.lower())))