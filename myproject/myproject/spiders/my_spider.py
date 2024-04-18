import scrapy


class MySpiderSpider(scrapy.Spider):
    name = "my_spider"
    start_urls = ["https://www.zyte.com/blog/"]

    def parse(self, response):
        for text in response.css('div.post-header h2 a::text').getall() :
            yield {
                'title' : text
            }
