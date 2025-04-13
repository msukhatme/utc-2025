import argparse
from utcxchangelib.xchange_client import XChangeClient, Side
from utcxchangelib.xchange_client import SWAP_MAP
import asyncio
import logging
import math
from math import log, sqrt
from scipy.stats import norm
import numpy as np
from collections import deque
import json
import os
import time

class Case1Bot(XChangeClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # General initializations
        self.market_values = {}
        self.theo_values = {}
        self.current_timestamp = 0
        self.price_history = {sym: deque(maxlen=100) for sym in ["APT", "DLR", "MKJ", "AKAV", "AKIM"]}  # Track recent mid-prices for momentum/volatility calculations (rolling window)
        self.trade_history = {sym: deque(maxlen=20) for sym in ["APT", "DLR", "MKJ", "AKAV", "AKIM"]}   # Track recent trade imbalances for each symbol (to gauge market aggressiveness)
        self.last_best_bid = {} # Store last known best bid for each symbol (for trade imbalance analysis)
        self.last_best_ask = {} # Store last known best ask for each symbol (for trade imbalance analysis)

        self.order_semaphore = asyncio.Semaphore(1)

        self.reload_config()

        # Set constants
        self.fee = 5            # fee for creating/redeeming AKAV (given)
        #self.margin = 1         # margin around market value for our quotes (chosen)
        self.risk_limits = {    # full risk limits (given)
            "max_order_size": 40,
            "max_open_orders": 50,
            "max_outstanding_volume": 120,
            "max_absolute_position": 200
        }

        # APT initializations
        self.apt_PE_ratio = 10  # constant P/E for APT earnings updates (given)

        # DLR initializations
        #self.DLR_init_theo = 5000
        #self.DLR_init_sigs = 0
        #self.theo_values["DLR"] = self.DLR_init_theo
        #self.dlr_signatures = self.DLR_init_sigs

        # MKJ initializations
        #self.MKJ_pos_kw = ["boom", "bull", "bullish", "long","goes viral", "gains popularity", "are buying", "bet big", "breaking: investors", "could transform", "next big", "meme coin", "memecoin", "quantum", "quantum computing", "astrology", "tarot", "time-traveling", "time travel", "time-travel", "flock", "elon", "elon tweets", "parallel universes", "future of global finance", "hidden patterns", "dream interpretation"],
        #self.MKJ_neg_kw = ["bear", "bearish", "market unchanged", "short the moon", "inflation", "hacker", "betting against", "short", "short squeeze", "bust"]
        #self.MKJ_news_delta = 0.05
        #self.MKJ_threshold = 20
        #self.MKJ_min_qty = 10
        #self.MKJ_max_qty = 20

        # AKAV initializations
        #self.max_swap_qty = 3

        # AKIM initializations
        #self.AKIM_flatten_time = 85
        #self.hedge_pos_factor = 0.7
        #self.hedge_qty_factor = 0.5

        # Filter initializations
        #self.filter_lower_bound_factor = 0.8
        #self.filter_upper_bound_factor = 1.2

        # Manage profit initializations
        #self.profit_threshold = 0.2      # 20%
        #self.exit_fraction = 0.3         # Exit 30% of position when profit condition met

        # Rolling volatility and optimal quotes initializations
        #self.rolling_vol_min_samples = 10
        #self.rolling_vol_target_window = 50
        #self.rolling_vol_default_vol = 10.0
        #self.rolling_vol_widen_factor = 1.5
        #self.optimal_risk_aversion = 0.5
        #self.optimal_T = 1.0
        #self.optimal_k = 1.0

        # TESTING
        # For tracking average entry price per asset for profit management
        self.avg_entry = {}   # e.g. {"DLR": average entry price}
        self.entry_qty = {}   # corresponding total quantity used in the average (absolute value)

        # MARKET VALUES LOGGER
        self.mv_logger = logging.getLogger("market_values_logger")
        if not self.mv_logger.handlers:
            mv_handler = logging.FileHandler("market_values.log", mode='w')
            mv_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            mv_handler.setFormatter(mv_formatter)
            self.mv_logger.addHandler(mv_handler)
            self.mv_logger.setLevel(logging.INFO)
        self.mv_logger.propagate = False

        # THEO VALUES LOGGER
        self.tv_logger = logging.getLogger("theo_values_logger")
        if not self.tv_logger.handlers:
            tv_handler = logging.FileHandler("theo_values.log", mode='w')
            tv_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            tv_handler.setFormatter(tv_formatter)
            self.tv_logger.addHandler(tv_handler)
            self.tv_logger.setLevel(logging.INFO)
        self.tv_logger.propagate = False

        # ARB LOGGER
        self.arb_logger = logging.getLogger("arb_logger")
        if not self.arb_logger.handlers:
            arb_handler = logging.FileHandler("arb.log", mode='w')
            arb_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            arb_handler.setFormatter(arb_formatter)
            self.arb_logger.addHandler(arb_handler)
            self.arb_logger.setLevel(logging.INFO)
        self.arb_logger.propagate = False

        # NEWS LOGGER
        self.news_logger = logging.getLogger("news_logger")
        if not self.news_logger.handlers:
            news_handler = logging.FileHandler("news.log", mode='w')
            news_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            news_handler.setFormatter(news_formatter)
            self.news_logger.addHandler(news_handler)
            self.news_logger.setLevel(logging.INFO)
        self.news_logger.propagate = False

        # ORDER LOGGER
        self.order_logger = logging.getLogger("order_logger")
        if not self.order_logger.handlers:
            order_handler = logging.FileHandler("orders.log", mode='w')
            order_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            order_handler.setFormatter(order_formatter)
            self.order_logger.addHandler(order_handler)
            self.order_logger.setLevel(logging.INFO)
        self.order_logger.propagate = False

        # RISK LOGGER
        self.risk_logger = logging.getLogger("risk_logger")
        if not self.risk_logger.handlers:
            risk_handler = logging.FileHandler("risk.log", mode='w')
            risk_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            risk_handler.setFormatter(risk_formatter)
            self.risk_logger.addHandler(risk_handler)
            self.risk_logger.setLevel(logging.INFO)
        self.risk_logger.propagate = False

    def reload_config(self):
        try:
            with open("config.json", "r") as f:
                cfg = json.load(f)
        except Exception as e:
            print("Error loading config.json:", e)
            return
        # Save configuration in instance variables:
        self.margin = cfg.get("margin", 1)
        #self.DLR_init_theo = cfg.get("DLR_init_theo", 5000)
        #self.DLR_init_sigs = cfg.get("DLR_init_sigs", 0)
        self.filter_lower_bound_factor = cfg.get("filter_lower_bound_factor", 0.8)
        self.filter_upper_bound_factor = cfg.get("filter_upper_bound_factor", 1.2)
        self.profit_threshold = cfg.get("profit_threshold", 0.2)
        self.exit_fraction = cfg.get("exit_fraction", 0.3)
        self.MKJ_pos_kw = cfg.get("MKJ_pos_kw", ["boom", "bull", "bullish", "long","goes viral", "gains popularity", "are buying", "bet big", "breaking: investors", "could transform", "next big", "meme coin", "memecoin", "quantum", "quantum computing", "astrology", "tarot", "time-traveling", "time travel", "time-travel", "flock", "elon", "elon tweets", "parallel universes", "future of global finance", "hidden patterns", "dream interpretation"])
        self.MKJ_neg_kw = cfg.get("MKJ_neg_kw", ["bear", "bearish", "market unchanged", "short the moon", "inflation", "hacker", "betting against", "short", "short squeeze", "bust"])
        self.MKJ_news_delta = cfg.get("MKJ_news_delta", 0.05)
        self.MKJ_threshold = cfg.get("MKJ_threshold", 20)
        self.MKJ_min_qty = cfg.get("MKJ_min_qty", 10)
        self.MKJ_max_qty = cfg.get("MKJ_max_qty", 20)
        self.AKIM_flatten_time = cfg.get("AKIM_flatten_time", 85)
        self.hedge_pos_factor = cfg.get("hedge_pos_factor", 0.7)
        self.hedge_qty_factor = cfg.get("hedge_qty_factor", 0.5)
        self.max_swap_qty = cfg.get("max_swap_qty", 3)
        self.rolling_vol_min_samples = cfg.get("rolling_vol_min_samples", 10)
        self.rolling_vol_target_window = cfg.get("rolling_vol_target_window", 50)
        self.rolling_vol_default_vol = cfg.get("rolling_vol_default_vol", 10.0)
        self.rolling_vol_widen_factor = cfg.get("rolling_vol_widen_factor", 1.5)
        self.optimal_risk_aversion = cfg.get("optimal_risk_aversion", 0.5)
        self.optimal_T = cfg.get("optimal_T", 1.0)
        self.optimal_k = cfg.get("optimal_k", 1.0)
        # You can print or log to confirm the load.
        print("Configuration reloaded at", time.strftime("%X"), cfg)

    async def update_config_task(self):
        """
        Periodically reloads configuration from config.json.
        """
        last_mod_time = os.path.getmtime("config.json")
        while True:
            try:
                new_mod_time = os.path.getmtime("config.json")
                if new_mod_time != last_mod_time:
                    self.reload_config()
                    last_mod_time = new_mod_time
            except Exception as e:
                print("Error checking config.json modification time:", e)
            await asyncio.sleep(0.5)

    # TESTING FILTER
    def filter_order_book(self, symbol: str):
        """
        Return filtered order book data for the given symbol.
        Orders with nonpositive quantity are discarded.
        If a previous market value is available, only orders priced within 80%-120%
        of that mid-price are kept.
        """
        book = self.order_books.get(symbol)
        if not book:
            return {}, {}
        prev_mid = self.market_values.get(symbol, None)
        if prev_mid is not None:
            lower_bound = int(prev_mid * self.filter_lower_bound_factor)
            upper_bound = int(prev_mid * self.filter_upper_bound_factor)
        else:
            # No previous mid available: accept all orders with positive quantities.
            lower_bound = 0
            upper_bound = float('inf')
        filtered_bids = {int(price): qty for price, qty in book.bids.items()
                        if qty > 0 and lower_bound <= int(price) <= upper_bound}
        filtered_asks = {int(price): qty for price, qty in book.asks.items()
                        if qty > 0 and lower_bound <= int(price) <= upper_bound}
        return filtered_bids, filtered_asks

    # TESTING
    def update_avg_entry_price(self, symbol: str, qty: int, price: int, side: str):
        """
        Maintain a weighted average cost for the open position.
        For orders that increase the position (same direction), update the weighted average.
        For orders reducing the position, leave the average unchanged (representing realized profit)
        unless the position is fully closed or reversed.
        """
        current_net = self.positions.get(symbol, 0)
        # Compute the previous net by “undoing” this fill.
        previous_net = current_net - qty if side == "buy" else current_net + qty
        # Case 1: Either there was no existing position or it was flat.
        if symbol not in self.avg_entry or self.avg_entry.get(symbol) is None or previous_net == 0:
            # If the fill exactly zeroes our previous position, then we consider the position closed.
            if current_net == 0:
                self.avg_entry[symbol] = None
                self.entry_qty[symbol] = 0
            else:
                # Otherwise, set the new cost basis to the fill price.
                self.avg_entry[symbol] = price
                self.entry_qty[symbol] = abs(current_net)
        else:
            # Check if the new fill is in the same direction as the previous position.
            if (side == "buy" and previous_net > 0) or (side == "sell" and previous_net < 0):
                # No reversal: If increasing the position, update the weighted average.
                if (side == "buy" and current_net > previous_net) or (side == "sell" and current_net < previous_net):
                    old_qty = self.entry_qty.get(symbol, 0)
                    new_total = old_qty + abs(qty)
                    new_avg = int((self.avg_entry[symbol] * old_qty + price * abs(qty)) / new_total)
                    self.avg_entry[symbol] = new_avg
                    self.entry_qty[symbol] = new_total
                else:
                    # Partial reduction without reversal: Keep the same average.
                    self.entry_qty[symbol] = abs(current_net)
            else:
                # Reversal detected: previous position and new position have opposite signs.
                # For example, if we were long and this sell not only closes but goes net short.
                if current_net == 0:
                    self.avg_entry[symbol] = None
                    self.entry_qty[symbol] = 0
                else:
                    # For the reversed part, the new average is the fill price.
                    self.avg_entry[symbol] = price
                    self.entry_qty[symbol] = abs(current_net)

    # TESTING
    async def bot_handle_order_fill(self, order_id: str, qty: int, price: int):
        self.order_logger.info(f"Order filled: {order_id} | {qty} @ {price}")
        order_details = self.open_orders.get(order_id)
        if order_details is not None:
            symbol = order_details[0].symbol
            # Update average entry price based on fill info.
            # Determine side from order_details – convert enum back to "buy"/"sell".
            side = "buy" if order_details[0].side == 1 else "sell"
            self.update_avg_entry_price(symbol, qty, price, side)
            # Removed the call to bot_handle_trade_msg to avoid double counting.
            # (Trade flow updates will be handled by the public trade message.)
            ###await self.bot_handle_trade_msg(symbol, price, qty)

    # TESTING
    async def manage_profit(self):
        """
        Periodically check each asset for profit levels.
        If a position (long/short) is in profit by more than a target percentage (default 0.2%),
        exit a fraction of the position (here 30%). Then, if the theoretical value still supports
        the original signal (e.g. undervalued for longs), re-enter by buying back that fraction.
        """
        profit_threshold = self.profit_threshold    # 20%
        exit_fraction = self.exit_fraction          # Exit 30% of position when profit condition met
        while True:
            for symbol in self.positions:
                if symbol == "cash":
                    continue
                net_qty = self.positions.get(symbol, 0)
                if net_qty == 0:
                    continue
                avg_price = self.avg_entry.get(symbol, None)
                if avg_price is None:
                    continue
                current_market = self.market_values.get(symbol, None)
                if current_market is None:
                    continue
                # For long positions: profit if current_market > avg_price*(1 + threshold)
                if net_qty > 0:
                    profit_pct = (current_market - avg_price) / avg_price
                    if profit_pct > profit_threshold:
                        exit_qty = max(1, int(exit_fraction * net_qty))
                        self.order_logger.info(f"[Profit Take] Long {symbol}: market={current_market}, avg={avg_price}, profit_pct={profit_pct:.4f} -> exit {exit_qty}")
                        # Use current best bid as limit price (if available)
                        f_bids, f_asks = self.filter_order_book(symbol)
                        if f_bids:
                            best_bid = max(f_bids.keys())
                        else:
                            best_bid = current_market
                        async with self.order_semaphore:
                            await self.safe_place_order(symbol, qty=exit_qty, side="sell", px=best_bid)
                        # Re-entry: if theoretical signal still holds (i.e. theo > market + margin)
                        theo = self.theo_values.get(symbol, None)
                        if theo is not None and theo > current_market + self.margin:
                            self.order_logger.info(f"[Re-Entry] Long {symbol}: conditions persist (theo={theo}, market={current_market}); re-enter {exit_qty}")
                            if f_asks:
                                best_ask = min(f_asks.keys())
                            else:
                                best_ask = current_market
                            async with self.order_semaphore:
                                await self.safe_place_order(symbol, qty=exit_qty, side="buy", px=best_ask)
                # For short positions: profit if current_market < avg_price*(1 - threshold)
                elif net_qty < 0:
                    profit_pct = (avg_price - current_market) / avg_price
                    if profit_pct > profit_threshold:
                        exit_qty = max(1, int(exit_fraction * abs(net_qty)))
                        self.order_logger.info(f"[Profit Take] Short {symbol}: market={current_market}, avg={avg_price}, profit_pct={profit_pct:.4f} -> exit {exit_qty}")
                        # Use current best ask as limit price (if available)
                        f_bids, f_asks = self.filter_order_book(symbol)
                        if f_asks:
                            best_ask = min(f_asks.keys())
                        else:
                            best_ask = current_market
                        async with self.order_semaphore:
                            await self.safe_place_order(symbol, qty=exit_qty, side="buy", px=best_ask)
                        # Re-entry: if theoretical signal still supports a short (i.e. theo < market - margin)
                        theo = self.theo_values.get(symbol, None)
                        if theo is not None and theo < current_market - self.margin:
                            self.order_logger.info(f"[Re-Entry] Short {symbol}: conditions persist (theo={theo}, market={current_market}); re-enter {exit_qty}")
                            if f_bids:
                                best_bid = max(f_bids.keys())
                            else:
                                best_bid = current_market
                            async with self.order_semaphore:
                                await self.safe_place_order(symbol, qty=exit_qty, side="sell", px=best_bid)
            await asyncio.sleep(0.2)

    # TESTING
    async def update_MKJ_theo(self, content: str):
        """
        Calculate MKJ theoretical value from unstructured news.
        Now, sentiment is measured by count of positive/negative keyword hits.
        A multiplier based on sentiment strength adjusts the target price change and order size.
        """
        sentiment_score = 0
        positive_keywords = self.MKJ_pos_kw
        negative_keywords = self.MKJ_neg_kw

        for kw in positive_keywords:
            if kw in content:
                sentiment_score += 1
        for kw in negative_keywords:
            if kw in content:
                sentiment_score -= 1

        current_mid = self.market_values.get("MKJ", None)
        if current_mid is None:
            return
        # Base delta: 5% of current price, at least 1 tick.
        delta = max(1, int(self.MKJ_news_delta * current_mid))
        if sentiment_score > 0:
            multiplier = 1 + 0.5 * (sentiment_score - 1)
            new_price = current_mid + int(delta * multiplier)
            self.theo_values["MKJ"] = new_price
            self.tv_logger.info(f"Positive sentiment news -> raising MKJ theo to {new_price} (multiplier {multiplier:.2f})")
            threshold = self.MKJ_threshold
            if new_price > current_mid + threshold:
                order_qty = self.MKJ_min_qty if sentiment_score <= 1 else self.MKJ_max_qty
                self.order_logger.info(f"[News Reaction] MKJ bullish; BUYing {order_qty} MKJ (news: '{content[:40]}...').")
                async with self.order_semaphore:
                    await self.safe_place_order("MKJ", qty=order_qty, side="buy")
        elif sentiment_score < 0:
            multiplier = 1 + 0.5 * (abs(sentiment_score) - 1)
            new_price = max(100, current_mid - int(delta * multiplier))
            self.theo_values["MKJ"] = new_price
            self.tv_logger.info(f"Negative sentiment news -> lowering MKJ theo to {new_price} (multiplier {multiplier:.2f})")
            threshold = self.MKJ_threshold
            if new_price < current_mid - threshold:
                order_qty = self.MKJ_min_qty if abs(sentiment_score) <= 1 else self.MKJ_max_qty
                self.order_logger.info(f"[News Reaction] MKJ bearish; SELLing {order_qty} MKJ (news: '{content[:40]}...').")
                async with self.order_semaphore:
                    await self.safe_place_order("MKJ", qty=order_qty, side="sell")
        else:
            self.news_logger.info(f"[News] No significant sentiment for MKJ; no action taken.")

    # TESTING
    async def cancel_orders_for_asset(self, asset: str):
        """
        Cancel all outstanding orders for the given asset.
        This helps avoid risk from outdated orders when news changes the theoretical view.
        """
        for order_id, order_entry in list(self.open_orders.items()):
            if order_entry[0].symbol == asset:
                try:
                    await self.cancel_order(order_id)
                    self.order_logger.info(f"[Cancel Orders] Canceled order {order_id} for asset {asset} due to news update.")
                except Exception as e:
                    self.order_logger.warning(f"[Cancel Orders] Failed to cancel order {order_id} for {asset}: {e}")
    # TESTING
    async def manage_risk(self):
        """
        Monitor effective positions (filled positions plus open orders) and hedge using AKIM to manage risk.
        Also, near the end of the trading day, flatten AKIM positions.
        """
        while True:
            try:
                # FIRST, compute effective positions by adding filled positions and pending order exposure.
                effective_positions = {}
                # Start with the filled positions.
                for asset, pos in self.positions.items():
                    if asset == "cash":
                        continue
                    effective_positions[asset] = pos
                # Add in the effect of open orders.
                for order_id, order_entry in self.open_orders.items():
                    order = order_entry[0]
                    qty = order_entry[1]
                    asset = order.symbol
                    if asset == "cash":
                        continue
                    # For a buy order, add quantity; for a sell order, subtract.
                    if order.side == Side.BUY:
                        effective_positions[asset] = effective_positions.get(asset, 0) + qty
                    else:
                        effective_positions[asset] = effective_positions.get(asset, 0) - qty

                # END-OF-DAY HEDGING for AKIM remains unchanged:
                if self.current_timestamp:
                    current_time_sec = self.current_timestamp / 5.0
                    if current_time_sec % 90 >= self.AKIM_flatten_time:
                        # Use the effective position for AKIM.
                        akim_eff = effective_positions.get("AKIM", 0)
                        if akim_eff != 0:
                            akim_bids, akim_asks = self.filter_order_book("AKIM")
                            if akim_eff > 0:
                                order_price = max(akim_bids.keys()) if akim_bids else self.market_values.get("AKIM", 0)
                                side = "sell"
                            else:
                                order_price = min(akim_asks.keys()) if akim_asks else self.market_values.get("AKIM", 0)
                                side = "buy"
                            self.risk_logger.info(f"[EOD Hedge] Flattening effective AKIM pos {akim_eff} with {side.upper()} order at {order_price}.")
                            async with self.order_semaphore:
                                await self.safe_place_order("AKIM", qty=abs(akim_eff), side=side, px=order_price, override=True)

                # Now, for each non-cash asset, hedge based on effective exposure.
                for asset, eff_pos in effective_positions.items():
                    # Skip cash.
                    if asset == "cash":
                        continue
                    # Use the absolute risk limit from configuration.
                    limit = self.risk_limits["max_absolute_position"]
                    # Define a threshold to trigger hedging (e.g. 85% of limit)
                    hedge_threshold = 0.85 * limit
                    if abs(eff_pos) > hedge_threshold and asset in ["APT", "DLR", "MKJ", "AKAV"]:
                        excess = abs(eff_pos) - hedge_threshold
                        # Hedge only a fraction (e.g. 50%) of the excess.
                        hedge_qty = int(excess * self.hedge_qty_factor)
                        if hedge_qty > 0:
                            if eff_pos > 0:
                                self.risk_logger.info(f"[Risk Management] {asset} effective pos {eff_pos} high; pre-hedging SELL {hedge_qty} AKIM.")
                                async with self.order_semaphore:
                                    await self.safe_place_order("AKIM", qty=hedge_qty, side="sell")
                            elif eff_pos < 0:
                                self.risk_logger.info(f"[Risk Management] {asset} effective pos {eff_pos} high; pre-hedging BUY {hedge_qty} AKIM.")
                                async with self.order_semaphore:
                                    await self.safe_place_order("AKIM", qty=hedge_qty, side="buy")
                    # If effective exposure exceeds the hard limit, cancel orders and hedge the remaining excess.
                    if abs(eff_pos) > limit:
                        self.risk_logger.info(f"[Risk Management] {asset} effective pos {eff_pos} exceeds limit; cancelling orders and hedging.")
                        for order_id, order_entry in list(self.open_orders.items()):
                            if order_entry[0].symbol == asset:
                                try:
                                    await self.cancel_order(order_id)
                                except Exception as e:
                                    self.risk_logger.info(f"[Risk Management] Error cancelling order {order_id}: {e}")
                        # Hedge the excess beyond the hard limit.
                        hedge_excess = abs(eff_pos) - limit
                        if asset in ["APT", "DLR", "MKJ", "AKAV"]:
                            if eff_pos > 0:
                                self.risk_logger.info(f"[Risk Management] Hedging long {asset} by BUYING {hedge_excess} AKIM.")
                                async with self.order_semaphore:
                                    await self.safe_place_order("AKIM", qty=hedge_excess, side="buy", override=True)
                            elif eff_pos < 0:
                                self.risk_logger.info(f"[Risk Management] Hedging short {asset} by SELLING {hedge_excess} AKIM.")
                                async with self.order_semaphore:    
                                    await self.safe_place_order("AKIM", qty=hedge_excess, side="sell", override=True)
            except Exception as e:
                self.risk_logger.error(f"[Manage Risk Error] {e}")
            await asyncio.sleep(0.2)

    async def update_market_values(self):
        """
        Wait 0.6 seconds for market calibration, then every 0.01 seconds update the market value 
        for each asset based solely on the current order books. For each symbol, if 
        both bids and asks exist, set the market value to the mid-price.
        """
        await asyncio.sleep(0.6)
        while True:
            try:
                for symbol in ["APT", "DLR", "MKJ", "AKAV", "AKIM"]:
                    filtered_bids, filtered_asks = self.filter_order_book(symbol)
                    if filtered_bids and filtered_asks:
                        best_bid = max(filtered_bids.keys())
                        best_ask = min(filtered_asks.keys())
                        mid = (best_bid + best_ask) // 2
                        self.market_values[symbol] = mid
                        self.mv_logger.info(f"Updated {symbol} to {mid} using filtered data")
                    else:
                        self.mv_logger.warning(f"{symbol} order book has no valid bids/asks after filtering")
            except Exception as e:
                self.mv_logger.error(f"[Update Market Values Error] {e}")
            await asyncio.sleep(0.2)

    async def update_APT_theo(self, earnings: float):
        """
        Calculate APT theoretical value.
        """
        self.theo_values["APT"] = int(earnings * self.apt_PE_ratio)
        self.tv_logger.info(f"Updated APT theo to {self.theo_values["APT"]} (earnings={earnings})")
    
    async def update_DLR_theo(self, cumulative: int, timestamp: int):
        """
        Calculate DLR theoretical value.
        """
        # If no petition news has arrived yet (or cumulative is 0), do not update theo.
        if cumulative <= 0:
            return
        seconds = timestamp // 5
        total_updates = 50
        events_elapsed = int(seconds // 15) - int(seconds // 90)
        n = total_updates - events_elapsed
        if n < 0:
            n = 0
        if cumulative <= 0:
            prob = 0.50
        elif n == 0:
            prob = 1.0 if cumulative >= 100000 else 0.0
        else:
            alpha = 1.0630449594499
            sigma = 0.006
            mu = math.log(cumulative) + n * math.log(alpha)
            sigma_eff = math.sqrt(n) * sigma
            try:
                z = (math.log(100000) - mu) / sigma_eff
            except ZeroDivisionError:
                z = float('inf')
            prob = 1 - (0.5 * (1 + math.erf(z / math.sqrt(2))))
        self.theo_values["DLR"] = int(prob * 10000)
        self.tv_logger.info(f"Updated DLR theo to {self.theo_values["DLR"]} (cumulative={cumulative}, prob={prob:.2f})")

    """
    async def update_MKJ_theo(self, content: str):
        
        ###Calculate MKJ theoretical value.
        
        sentiment_score = 0
        positive_keywords = ["goes viral", "gains popularity", "are buying", "bet big", "breaking: investors", "could transform", "next big"]
        negative_keywords = ["market unchanged", "short the moon", "inflation"]

        for kw in positive_keywords:
            if kw in content:
                sentiment_score += 1
        for kw in negative_keywords:
            if kw in content:
                sentiment_score -= 1

        current_mid = self.market_values["MKJ"]
        # e.g. raise theo value by 5% or a fixed amount
        delta = max(1, int(0.05 * current_mid))  # 5% of current price (at least 1)

        if sentiment_score > 0:
            new_price = current_mid + delta
            self.theo_values["MKJ"] = new_price
            self.tv_logger.info(f"Positive sentiment news -> raising MKJ theo value to {new_price}")
            # Aggressively buy if market is below new theo value
            threshold = 2  # smaller threshold for immediate action on strong news
            if new_price > current_mid + threshold:
                self.order_logger.info(f"[News Reaction] MKJ sentiment bullish; BUYing 10 MKJ (news: '{content[:40]}...').")
                async with self.order_semaphore:
                    await self.safe_place_order("MKJ", qty=10, side="buy")
        elif sentiment_score < 0:
            new_price = max(100, current_mid - delta)  # don't go below 1
            self.theo_values["MKJ"] = new_price
            self.tv_logger.info(f"Negative/neutral news -> lowering MKJ theo value to {new_price}")
            # Aggressively sell if market is above new theo value
            threshold = 2
            if new_price < current_mid - threshold:
                self.order_logger.info(f"[News Reaction] MKJ sentiment bearish; SELLing 10 MKJ (news: '{content[:40]}...').")
                async with self.order_semaphore:
                    await self.safe_place_order("MKJ", qty=10, side="sell")
        else:
            # Neutral or indeterminate news – no action
            self.news_logger.info(f"[News] No significant sentiment detected for MKJ; no trading action.")
    """
            
    async def update_AKAV_theo(self):
        """
        Calculate AKAV theoretical value.
        """
        if all(sym in self.theo_values for sym in ["APT", "DLR", "MKJ"]):
            computed_akav = self.theo_values["APT"] + self.theo_values["DLR"] + self.theo_values["MKJ"]
            self.theo_values["AKAV"] = computed_akav
            self.tv_logger.info(f"Computed underlying AKAV = {computed_akav}")

    async def update_AKIM_theo(self):
        """
        Calculate AKIM theoretical value.
        """
        # Initialize day start references if not already
        if not hasattr(self, "akav_day_start"):
            if "AKAV" in self.market_values and "AKIM" in self.market_values:
                self.akav_day_start = self.market_values["AKAV"]
                self.akim_day_start = self.market_values["AKIM"]
                self.current_day = 0
        # Detect new trading day based on timestamp (90s per day)
        if self.current_timestamp:
            cur_day = int((self.current_timestamp / 5.0) // 90)  # 5 ticks = 1 sec
            if hasattr(self, "current_day") and cur_day != self.current_day:
                # New day: reset reference prices
                if "AKAV" in self.market_values and "AKIM" in self.market_values:
                    self.akav_day_start = self.market_values["AKAV"]
                    self.akim_day_start = self.market_values["AKIM"]
                self.current_day = cur_day
        # Compute theoretical AKIM as inverse daily return of AKAV
        if hasattr(self, "akav_day_start") and hasattr(self, "akim_day_start"):
            if "AKAV" in self.market_values and self.akav_day_start > 0:
                akav_cur = self.market_values["AKAV"]
                r = (akav_cur / self.akav_day_start) - 1.0   # AKAV return since open
                akim_theo = int(self.akim_day_start * (1 - r))
                self.theo_values["AKIM"] = akim_theo
                self.tv_logger.info(f"Updated AKIM theo to {akim_theo} "
                                    f"(AKAV daily return={r:.4f})")

    async def check_etf_arbitrage(self):
        """
        Checks for arbitrage opportunities between ETF AKAV and its underlying stocks (APT, DLR, MKJ).
        
        It does so by comparing:
          - The composite bid price of the underlying stocks (sum of their best bids)
          - The composite ask price of the underlying stocks (sum of their best asks)
        
        Then it compares these with AKAV’s best bid/ask (adjusted for fees).
          - If AKAV is undervalued (AKAV ask + fee_fromAKAV < composite bid – margin),
            execute a redemption arbitrage (buy AKAV, redeem into underlying, then sell the underlyings).
          - If AKAV is overvalued (composite ask + fee_toAKAV < AKAV bid – margin),
            execute a creation arbitrage (buy the underlyings, create AKAV, then sell AKAV).
          
        Risk limits are checked (if any underlying is at a risk limit, the function aborts).
        """
        while True:
            try:
                # For each asset, get filtered order books
                apt_bids, apt_asks = self.filter_order_book("APT")
                dlr_bids, dlr_asks = self.filter_order_book("DLR")
                mkj_bids, mkj_asks = self.filter_order_book("MKJ")
                akav_bids, akav_asks = self.filter_order_book("AKAV")
                if not (apt_bids and apt_asks and dlr_bids and dlr_asks and mkj_bids and mkj_asks and akav_bids and akav_asks):
                    await asyncio.sleep(0.2)
                    continue

                apt_best_bid = max(apt_bids.keys())
                apt_best_ask = min(apt_asks.keys())
                dlr_best_bid = max(dlr_bids.keys())
                dlr_best_ask = min(dlr_asks.keys())
                mkj_best_bid = max(mkj_bids.keys())
                mkj_best_ask = min(mkj_asks.keys())
                akav_best_bid = max(akav_bids.keys())
                akav_best_ask = min(akav_asks.keys())

                # Composite prices for the underlying basket.
                composite_bid = apt_best_bid + dlr_best_bid + mkj_best_bid
                composite_ask = apt_best_ask + dlr_best_ask + mkj_best_ask

                # Define fees for swaps (creating/redeeming) as given.
                fee_toAKAV = self.fee
                fee_fromAKAV = self.fee

                # Log the current state
                self.arb_logger.info(
                    f"[Arb] Underlying bid sum={composite_bid}, ask sum={composite_ask}; "
                    f"AKAV bid={akav_best_bid}, ask={akav_best_ask}"
                )

                # Risk check: ensure that for each underlying we are not at our risk limit.
                max_pos = self.risk_limits["max_absolute_position"]
                if (self.positions.get("APT", 0) <= -max_pos or 
                    self.positions.get("DLR", 0) <= -max_pos or 
                    self.positions.get("MKJ", 0) <= -max_pos):
                    self.arb_logger.warning("[Arb] At short limit for an underlying; skipping AKAV redemption.")
                    await asyncio.sleep(0.2)
                    continue
                if (self.positions.get("APT", 0) >= max_pos or 
                    self.positions.get("DLR", 0) >= max_pos or 
                    self.positions.get("MKJ", 0) >= max_pos):
                    self.arb_logger.warning("[Arb] At long limit for an underlying; skipping AKAV creation.")
                    await asyncio.sleep(0.2)
                    continue

                # Arbitrage Condition 1: AKAV appears undervalued (redemption opportunity)
                if akav_best_ask + fee_fromAKAV < composite_bid - self.margin:
                    price_diff = composite_bid - (akav_best_ask + fee_fromAKAV)
                    self.arb_logger.info(f"[Arb] AKAV undervalued by ~{price_diff}; executing redemption arbitrage.")
                    # Determine how many swaps to perform (up to 3 if mispricing is large)
                    swaps_to_do = 1
                    if price_diff > 2 * self.margin:
                        swaps_to_do = min(self.max_swap_qty, 1 + price_diff // self.margin)
                    for i in range(int(swaps_to_do)):
                        async with self.order_semaphore:
                            # Buy 1 AKAV at its ask price.
                            await self.safe_place_order("AKAV", qty=1, side="buy", px=akav_best_ask)
                        # Request swap: redeem AKAV into its underlyings.
                        await self.place_swap_order('fromAKAV', qty=1)
                        # Sell the received underlying shares at their respective best bid prices.
                        async with self.order_semaphore:
                            await self.safe_place_order("APT", qty=1, side="sell", px=apt_best_bid)
                        async with self.order_semaphore:
                            await self.safe_place_order("DLR", qty=1, side="sell", px=dlr_best_bid)
                        async with self.order_semaphore:
                            await self.safe_place_order("MKJ", qty=1, side="sell", px=mkj_best_bid)
                        self.arb_logger.info(f"[Arb] Redemption iteration {i+1}: Executed swap and sold underlyings.")
                
                # Arbitrage Condition 2: AKAV appears overvalued (creation opportunity)
                elif composite_ask + fee_toAKAV < akav_best_bid - self.margin:
                    price_diff = (akav_best_bid - fee_toAKAV) - composite_ask
                    self.arb_logger.info(f"[Arb] AKAV overvalued by ~{price_diff}; executing creation arbitrage.")
                    swaps_to_do = 1
                    if price_diff > 2 * self.margin:
                        swaps_to_do = min(self.max_swap_qty, 1 + price_diff // self.margin)
                    for i in range(int(swaps_to_do)):
                        # Buy one of each underlying at their ask prices.
                        async with self.order_semaphore:
                            await self.safe_place_order("APT", qty=1, side="buy", px=apt_best_ask)
                        async with self.order_semaphore:
                            await self.safe_place_order("DLR", qty=1, side="buy", px=dlr_best_ask)
                        async with self.order_semaphore:
                            await self.safe_place_order("MKJ", qty=1, side="buy", px=mkj_best_ask)
                        # Swap underlyings for 1 AKAV (creation).
                        await self.place_swap_order('toAKAV', qty=1)
                        # Sell the created AKAV at its best bid price.
                        async with self.order_semaphore:
                            await self.safe_place_order("AKAV", qty=1, side="sell", px=akav_best_bid)
                        self.arb_logger.info(f"[Arb] Creation iteration {i+1}: Executed swap and sold AKAV.")
            except Exception as e:
                self.arb_logger.error(f"[Arb] Exception in check_etf_arbitrage: {e}")
            await asyncio.sleep(0.2)

    async def speculative(self):
        """
        Speculative trading based on comparison of theoretical value vs. market value.
        If theo > market value + margin, buy.
        If theo < market value - margin, sell.
        """
        while True:
            for symbol in ["APT", "DLR", "MKJ", "AKAV", "AKIM"]:
                f_bids, f_asks = self.filter_order_book(symbol)
                if symbol not in self.theo_values:
                    continue
                theo = self.theo_values[symbol]
                if f_bids and theo > max(f_bids.keys()) + self.margin:
                    async with self.order_semaphore:
                        await self.safe_place_order(symbol, qty=1, side="buy")
                if f_asks and theo < min(f_asks.keys()) - self.margin:
                    async with self.order_semaphore:
                        await self.safe_place_order(symbol, qty=1, side="sell")
            await asyncio.sleep(0.2)
    
    """
    async def manage_risk(self):

        #Monitor positions and hedge using AKIM to manage risk.

        while True:
            try:
                # End-of-day: flatten any AKIM position
                if self.current_timestamp:
                    current_time_sec = self.current_timestamp / 5.0
                    if current_time_sec % 90 >= 85:  # last 5 seconds of the day
                        akim_pos = self.positions.get("AKIM", 0)
                        if akim_pos != 0:
                            side = "sell" if akim_pos > 0 else "buy"
                            self.risk_logger.info(f"[EOD Hedge] Flattening AKIM position "
                                                f"{akim_pos} with {side.upper()} order.")
                            async with self.order_semaphore:
                                await self.safe_place_order("AKIM", qty=abs(akim_pos), side=side)
                # Check each asset's position against limits
                for asset, pos in self.positions.items():
                    if asset == "cash":
                        continue
                    limit = self.risk_limits["max_absolute_position"]
                    # Preemptive hedge if >70% limit (for underlying assets and AKAV)
                    if abs(pos) > 0.7 * limit and asset in ["APT", "DLR", "MKJ", "AKAV"]:
                        hedge_qty = int(abs(pos) - 0.5 * limit)
                        if hedge_qty > 0:
                            if pos > 0:
                                self.risk_logger.info(f"[Risk Management] {asset} pos {pos} high; "
                                                        f"pre-hedge SELL {hedge_qty} AKIM.")
                                async with self.order_semaphore:
                                    await self.safe_place_order("AKIM", qty=hedge_qty, side="sell")
                            elif pos < 0:
                                self.risk_logger.info(f"[Risk Management] {asset} pos {pos} high; "
                                                        f"pre-hedge BUY {hedge_qty} AKIM.")
                                async with self.order_semaphore:
                                    await self.safe_place_order("AKIM", qty=hedge_qty, side="buy")
                    # If position exceeds hard limit, cancel orders and fully hedge
                    if abs(pos) > limit:
                        self.risk_logger.info(f"[Risk Management] {asset} position {pos} "
                                                f"exceeds limit {limit}. Cancelling orders and hedging.")
                        # Cancel all open orders for this asset
                        for order_id, order in list(self.open_orders.items()):
                            if order[0].symbol == asset:
                                try:
                                    await self.cancel_order(order_id)
                                except Exception as e:
                                    self.risk_logger.info(f"[Risk Management] Error cancelling {order_id}: {e}")
                        # Hedge the excess exposure with AKIM
                        hedge_qty = abs(pos) - limit
                        if asset in ["APT", "DLR", "MKJ", "AKAV"]:
                            if pos > 0:
                                self.risk_logger.info(f"[Risk Management] Hedging long {asset} by BUYING {hedge_qty} AKIM.")
                                async with self.order_semaphore:
                                    await self.safe_place_order("AKIM", qty=hedge_qty, side="buy", override=True)
                            elif pos < 0:
                                self.risk_logger.info(f"[Risk Management] Hedging short {asset} by SELLING {hedge_qty} AKIM.")
                                async with self.order_semaphore:
                                    await self.safe_place_order("AKIM", qty=hedge_qty, side="sell", override=True)
            except Exception as e:
                self.risk_logger.error(f"[Manage Risk Error] {e}")
            await asyncio.sleep(0.2)
    """

    async def safe_place_order(self, symbol: str, qty: int, side: str, px: int = None, override: bool = False) -> bool:
        """
        Place an order only if the new order will keep the effective position within risk limits.
        The effective position is the sum of filled positions (self.positions)
        plus the net quantity of all outstanding orders for that asset.
        If not overriding, risk limits (max_order_size, max_open_orders,
        max_outstanding_volume, and max_absolute_position) are enforced.
        """
        # Compute the effective position for this symbol.
        effective = self.positions.get(symbol, 0)
        for order_id, order_entry in self.open_orders.items():
            order = order_entry[0]
            pending_qty = order_entry[1]
            if order.symbol == symbol:
                # Add pending quantities if it's a buy, subtract if sell.
                if order.side == Side.BUY:
                    effective += pending_qty
                else:
                    effective -= pending_qty

        # Calculate the change that would occur from this order.
        delta = qty if side == "buy" else -qty
        new_effective = effective + delta

        # Enforce risk limits (if not overriding) on effective position.
        if not override:
            if abs(new_effective) > self.risk_limits["max_absolute_position"]:
                self.order_logger.warning(
                    f"[Risk Limit] Effective position for {symbol} would go to {new_effective}, "
                    f"exceeding the limit of {self.risk_limits['max_absolute_position']}."
                )
                return False

            # You can also add checks for max_order_size, max_open_orders, and
            # max_outstanding_volume here as you already do.
            if qty > self.risk_limits["max_order_size"]:
                self.order_logger.warning(
                    f"[Risk Limit] Order qty {qty} for {symbol} exceeds the max order size "
                    f"{self.risk_limits['max_order_size']}."
                )
                await self.place_order(symbol, qty=self.risk_limits["max_order_size"], side=side, px=px)
                # Recursively try for the remaining quantity.
                await self.safe_place_order(symbol, qty=qty - self.risk_limits["max_order_size"], side=side, px=px)
                return True

            open_orders_for_symbol = [
                oid for oid, order in self.open_orders.items() if order[0].symbol == symbol
            ]
            if len(open_orders_for_symbol) >= self.risk_limits["max_open_orders"]:
                self.order_logger.warning(
                    f"[Risk Limit] Open orders for {symbol} exceeded {self.risk_limits['max_open_orders']}."
                )
                return False

            outstanding_vol = sum(order[1] for oid, order in self.open_orders.items() if order[0].symbol == symbol)
            if outstanding_vol + qty > self.risk_limits["max_outstanding_volume"]:
                self.order_logger.warning(
                    f"[Risk Limit] Outstanding volume for {symbol} would exceed limit "
                    f"{self.risk_limits['max_outstanding_volume']}."
                )
                return False

        self.order_logger.info(
            f"[Safe Order] Placing {side.upper()} order for {qty} {symbol} at {px if px is not None else 'market'} "
            f"(effective position after order: {new_effective})."
        )
        await self.place_order(symbol, qty=qty, side=side, px=px)
        return True

    async def bot_handle_news(self, news_release: dict):
        """
        Process an incoming news message.
        For structured news:
            - For APT: use earnings to update theoretical value.
            - For DLR: use petition data (cumulative signatures) to update theoretical value via update_DLR_theo.
        For unstructured news:
            - Adjust MKJ theoretical value based on simple sentiment analysis.
        Also, cancel outstanding orders for the affected asset to reduce risk.
        """
        timestamp = news_release.get("timestamp", 0)
        kind = news_release.get("kind", "")
        data = news_release.get("new_data", {})

        self.current_timestamp = timestamp
        self.news_logger.info("[News] " + " ---- ".join(f"{key}: {value}" for key, value in news_release.items()))

        if kind == "structured":
            subtype = data.get("structured_subtype", "")
            asset = data.get("asset", "")
            # Cancel any outstanding orders for the asset before processing news.
            await self.cancel_orders_for_asset(asset)
            # Process APT earnings news
            if subtype == "earnings" and asset == "APT":
                earnings = data.get("value", 0)
                await self.update_APT_theo(earnings)
                self.news_logger.info(f"[News] APT earnings update: earnings={earnings} -> theo value updated to {self.theo_values["APT"]}")
            # Process DLR petition news
            elif subtype == "petition" and asset == "DLR":
                cumulative = data.get("cumulative", 0)
                await self.update_DLR_theo(cumulative, timestamp)
                self.news_logger.info(f"[News] DLR petition update: cumulative={cumulative} -> theo value updated to {self.theo_values["DLR"]}")
            else:
                self.news_logger.error(f"[News] Received structured news for {asset} of type {subtype}")
        else:
            # Cancel any outstanding orders for MKJ before processing news.
            await self.cancel_orders_for_asset("MKJ")
            content = data.get("content", "")
            self.news_logger.info(f"[News] Unstructured news: {content}")
            content_lower = content.lower()
            await self.update_MKJ_theo(content_lower)

    async def bot_handle_book_update(self, symbol: str) -> None:
        """
        On each order book update, adjust our quotes for the given symbol.
        """
        if symbol not in self.theo_values:
            return  # skip if no theoretical value yet
        filtered_bids, filtered_asks = self.filter_order_book(symbol)
        # Record inside market and update price history
        best_bid = max(filtered_bids.keys(), default=None)
        best_ask = min(filtered_asks.keys(), default=None)
        self.last_best_bid[symbol] = best_bid
        self.last_best_ask[symbol] = best_ask
        mid_price = None
        if best_bid is not None and best_ask is not None:
            mid_price = (best_bid + best_ask) // 2
        if mid_price is not None:
            self.price_history[symbol].append(mid_price)
        # Compute our optimal quoting prices
        bid_price, ask_price = await self.compute_optimal_quotes(symbol)
        if bid_price is None or ask_price is None:
            return  # cannot quote yet
        order_size = 10 if symbol in ["APT", "DLR", "MKJ"] else 5  # smaller size for ETFs
        try:
            # Post a buy order if our bid is above the current best bid
            if best_bid is None or bid_price > best_bid:
                self.tv_logger.info(f"[{symbol}] Quoting BUY @ {bid_price} "
                                    f"(inv={self.positions.get(symbol,0)}, "
                                    f"σ={self.compute_rolling_volatility(self.price_history[symbol]):.2f})")
                async with self.order_semaphore:
                    await self.safe_place_order(symbol, qty=order_size, side="buy", px=bid_price)
            # Post a sell order if our ask is below the current best ask
            if best_ask is None or ask_price < best_ask:
                self.tv_logger.info(f"[{symbol}] Quoting SELL @ {ask_price} "
                                    f"(inv={self.positions.get(symbol,0)}, "
                                    f"σ={self.compute_rolling_volatility(self.price_history[symbol]):.2f})")
                async with self.order_semaphore:
                    await self.safe_place_order(symbol, qty=order_size, side="sell", px=ask_price)
        except Exception as e:
            self.risk_logger.error(f"[Book Update Error] {symbol}: {e}")

    async def bot_handle_trade_msg(self, symbol: str, price: int, qty: int):
        """
        Capture trade message to update recent trade imbalance (market aggressiveness indicator).
        If a trade executes at or above the last best ask, treat as aggressive buy (positive flow).
        If at or below the last best bid, treat as aggressive sell (negative flow).
        """
        last_bid = self.last_best_bid.get(symbol); last_ask = self.last_best_ask.get(symbol)
        if last_bid is None or last_ask is None:
            return  # Not enough info to determine aggressor
        if price >= last_ask:
            # Trade happened at ask or higher -> buyer-initiated (lifting the ask)
            self.trade_history[symbol].append(qty)
        elif price <= last_bid:
            # Trade at bid or lower -> seller-initiated (hitting the bid)
            self.trade_history[symbol].append(-qty)
        # (If the trade price is between the bid and ask, we do not count it as clear imbalance)

    """
    async def bot_handle_order_fill(self, order_id: str, qty: int, price: int):
        self.order_logger.info(f"Order filled: {order_id} | {qty} @ {price}")
        # Attempt to determine the symbol (assuming you store the order details in open_orders)
        order_details = self.open_orders.get(order_id)
        if order_details is not None:
            symbol = order_details[0].symbol
            # Update trade history via our new trade message handler.
            await self.bot_handle_trade_msg(symbol, price, qty)
    """
    
    async def handle_order_rejected(self, msg) -> None:
        """
        Override the parent's order rejected handler to safely remove the order.
        """
        self.order_logger.warning(f"[Order Rejected] Order ID {msg.id}: {msg.reason}")
        removed_order = self.open_orders.pop(msg.id, None)
        if removed_order is None:
            self.order_logger.warning(f"[Warning] Order id {msg.id} was not found in open_orders upon rejection.")

    def compute_rolling_volatility(self, prices: deque, min_samples: int = 10, target_window: int = 50, default_vol: float = 10.0, widen_factor: float = 1.5) -> float:
        """
        Estimate volatility from recent prices using rolling log-returns.
        """
        if len(prices) < 2:
            return default_vol
        # Compute log returns
        log_returns = [math.log(prices[i] / prices[i-1]) for i in range(1, len(prices)) if prices[i-1] > 0]
        # Limit to target window
        if len(log_returns) > target_window:
            log_returns = log_returns[-target_window:]
        # If too few samples, return std dev * widen_factor
        if len(log_returns) < min_samples:
            vol = np.std(log_returns, ddof=1) if len(log_returns) > 1 else default_vol
            return float(vol) * widen_factor
        else:
            return float(np.std(log_returns, ddof=1))

    async def compute_optimal_quotes(self, symbol: str) -> tuple[int, int]:
        """
        Compute optimal bid and ask for a symbol, adjusted for inventory, momentum, etc.
        """
        theo = self.theo_values.get(symbol)
        if theo is None:
            return (None, None)  # no theo value known yet
        inventory = self.positions.get(symbol, 0)
        # Get recent volatility and momentum
        prices = self.price_history[symbol]
        sigma = self.compute_rolling_volatility(prices)
        momentum = 0
        if len(prices) >= 5:
            momentum = prices[-1] - prices[-5]
        # Market conditions: best bid/ask and net trade flow
        best_bid = self.last_best_bid.get(symbol)
        best_ask = self.last_best_ask.get(symbol)
        spread = (best_ask - best_bid) if (best_bid is not None and best_ask is not None) else float('inf')
        net_flow = sum(self.trade_history[symbol])  # positive if buys > sells
        # Base reservation price with inventory risk adjustment
        risk_aversion = self.optimal_risk_aversion
        T = self.optimal_T
        k = self.optimal_k
        reservation_price = theo - (risk_aversion * (sigma ** 2) * T * inventory)
        # Momentum skew
        if momentum > 2:       # uptrend
            reservation_price += 1
        elif momentum < -2:    # downtrend
            reservation_price -= 1
        # Order flow skew (only if spread is very tight)
        if spread <= 2:
            if net_flow > 10:
                reservation_price += 1   # buying pressure
            elif net_flow < -10:
                reservation_price -= 1   # selling pressure
        # Optimal spread based on volatility and risk aversion
        total_spread = risk_aversion * (sigma ** 2) * T + (2 / risk_aversion) * math.log(1 + risk_aversion / k)
        half_spread = total_spread / 2.0
        optimal_bid = reservation_price - half_spread
        optimal_ask = reservation_price + half_spread
        # Round to nearest tick
        bid_px = int(optimal_bid)
        ask_px = int(optimal_ask)
        # Adjust to be inside the market if possible, without violating theo value
        if best_bid is not None and bid_px <= best_bid:
            bid_px = best_bid + 1 if best_bid + 1 < theo else bid_px
        if best_ask is not None and ask_px >= best_ask:
            ask_px = best_ask - 1 if best_ask - 1 > theo else ask_px
        return (bid_px, ask_px)

    async def start(self, user_interface: bool = False):
        asyncio.create_task(self.update_config_task())
        asyncio.create_task(self.update_market_values())
        asyncio.create_task(self.check_etf_arbitrage())
        asyncio.create_task(self.speculative())
        asyncio.create_task(self.manage_risk())
        asyncio.create_task(self.manage_profit())   # TESTING

        if user_interface:
            self.launch_user_interface()
            asyncio.create_task(self.handle_queued_messages())
        await self.connect()

async def main(user_interface: bool):
    #SERVER = '3.138.154.148:3333'
    SERVER = 'server.uchicagotradingcompetition25.com:3333'
    TEAM = 'chicago11'
    PASS = 'Ntzv6QfWb('
    bot = Case1Bot(SERVER,TEAM,PASS)
    await bot.start(user_interface)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script that connects client to exchange, runs algorithmic trading logic, and optionally deploys Phoenixhood"
    )

    parser.add_argument("--phoenixhood", required=False, default=False, type=bool, help="Starts phoenixhood API if true")
    args = parser.parse_args()

    user_interface = args.phoenixhood

    asyncio.run(main(user_interface))
