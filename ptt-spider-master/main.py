import asyncio
import itertools
import pprint
import json
from tqdm import tqdm

from ptt.core import PTTSpider


loop = asyncio.get_event_loop()


def coroutine_runner(coroutine):
    return loop.run_until_complete(coroutine)


if __name__ == '__main__':

    num_page = 100  # collect how many page
    for epoch in tqdm(range(0,3600,100)):
        with PTTSpider(board='HatePolitics', loop=loop) as spider:
            ''' get total page of current board '''
            total_pages = coroutine_runner(spider.get_total_page_num())

            ''' get pages of posts meta '''
            pages = range(total_pages - epoch, total_pages - epoch - num_page, -1)

            coros = asyncio.gather(
                *(spider.get_meta(page) for page in pages))

            metas = coroutine_runner(coros)
            metas = list(itertools.chain.from_iterable(metas))

            #get posts content 
            coros = asyncio.gather(
                *(spider.get_post(meta['link']) for meta in metas))

            posts = coroutine_runner(coros)

        with open(f"../politics_new/{epoch}.json","w+",encoding='UTF-8') as f :
            json.dump(posts,f,indent=2,ensure_ascii=False) 
        '''
        pprint.pprint(metas)
        print('total meta:', len(metas))



        pprint.pprint(posts)

        print('Total posts:', len(posts))
        '''