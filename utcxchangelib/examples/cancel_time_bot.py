from typing import Optional
import xchange_client
import asyncio
import time
import pandas as pd

class CancelLatencyTestClient(xchange_client.XChangeClient):

    def __init__(self, host: str, username: str, password: str):
        super().__init__(host, username, password, silent=True)
        self.cancel_times = []
        self.cancel_requested = False
        self.cancel_time = 0


    async def bot_handle_cancel_response(self, order_id: str, success: bool, error: Optional[str]) -> None:
        cancel_time = time.time() - self.cancel_time
        self.cancel_times.append(cancel_time)
        self.cancel_requested = False


    async def run_cancel_sim(self):
        print("starting cancel routine")
        await asyncio.sleep(3)
        while len(self.cancel_times) < 100:
            if self.cancel_requested:
                await asyncio.sleep(.1)
                continue
            await self.place_order("BRV",3, xchange_client.Side.BUY, 5)
            await self.cancel_order(list(self.open_orders.keys())[0])
            self.cancel_time = time.time()
            self.cancel_requested = True
        cancel_df = pd.DataFrame({'cancel_times':self.cancel_times})
        print("cancel times",self.cancel_times)
        print(cancel_df.describe())

    async def start(self):
        asyncio.create_task(self.run_cancel_sim())
        await self.connect()


async def main():
    # SERVER = '127.0.0.1:8000'
    SERVER = '13.58.49.87:3333'
    my_client = CancelLatencyTestClient(SERVER,"alice","alice_password")
    await my_client.start()
    return

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(main())



