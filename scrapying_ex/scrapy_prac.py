import scrapy

class ExampleSpider(scrapy.Spider) :
    name = "example_spider"
    start_urls = ['https://brand.naver.com/espoir/products/10123826918']

    def parse(self, response) :
        self.log('Visited %s' % response.url)
        page_title = response.css('title::text').extract_first().strip()
        yield{'Title' : page_title}