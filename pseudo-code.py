from time import sleep
from datetime import timedelta

# Configuration
config = {
    'exchanges' : ['Binance', 'OKX', 'ByBit', 'and some more'],
    'pairs' : ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT', 'SOL/USDT', 
              'AVAX/USDT', 'ADA/USDT', 'LTC/USDT', 'top 20 liquid pairs'],
    'risk_factor' : 0.01,           # Base risk per unit of signal
    'max_position_pct' : 0.10,      # Max position size as % of account equity
    'entry_threshold_factor' : 2.0, # Multiple of break-even threshold
    'max_slippage_bps': 1.5,        # Maximum acceptable slippage in basis points

    # Leverage for different market conditions
    'leverage_settings': {
        'low_vol': 7.0,
        'medium_vol': 5.0,
        'high_vol' : 3.0
    },

    'margin_buffer': 3.0,   # 300% minimum margin
    'buffer_triggers': {
        'warning': 0.70,    # Reduce by 20%
        'danger': 0.85,     # Reduce by 50%
        'critical': 0.95    # Close all positions
    },
    'execution_start': 30,  # Minutes before settlement time
    'execution_timeout': 3, # Minutes before we update the unfilled limit orders

    'calibration': {
        'signal_window': 90,         # 3 month window for signal normalization
        'vol_window': 90,            # 3 month window for volatility thresholds
        'break_even_window': 30,     # 1 month window for break-even calculations
        'signal_recalibrate': 30,    # Monthly recalibration for signals
        'volatility_recalibrate': 7, # Weekly recalibration for volatility
        'breakeven_recalibrate': 7,  # Weekly recalibration for break-even
    }
}


class FundingArbitrage:
    def __init__(self, config, account_equity):
        self.config = config
        self.account_equity = account_equity
        self.active_positions = {}

        self.last_calibration = {
            'signal': self.current_time(),
            'volatility': self.current_time(),
            'breakeven': self.current_time(),
        }

    def run_strategy(self):
        """Main strategy loop"""
        while True:
            next_funding = self.get_next_funding_event()

            # Approaching funding time: scan for trades
            if self.minutes_to_funding(next_funding) <= self.config['execution_start']:
                opportunities = self.scan_for_trades()
                if opportunities:
                    self.allocate_and_execute(opportunities)

            # Manage active positions
            self.manage_risk()

            # Run calibration
            self.calibrate_parameters()

            # Update system every 10 seconds
            sleep(10) # TODO questionable when nearing execution

    def scan_for_trades(self):
        """For each pair, scan all exchanges for opportunities and rank them by signal strength"""
        opportunities = []

        for pair in self.config['pairs']:
            # Get funding rates and open interest
            exchange_rates = {ex: self.get_funding_rates(ex, pair) for ex in self.config['exchanges']}
            open_interest = self.get_open_interest(pair)
            volatility = self.get_oi_volatility(pair)
            vol_regime = self.classify_volatility(pair, volatility)

            # Get median signal for normalization
            median_signal = self.get_median_signal(pair)

            # Check all exchange combinations
            for ex1, ex2 in self.get_exchange_combinations():
                funding_diff = exchange_rates[ex1] - exchange_rates[ex2]

                # Compute the break-even threshold
                break_even = self.compute_break_even(ex1, ex2, pair)
                min_threshold = break_even * self.config['entry_threshold_factor']

                # Skip if threshold is not met
                if abs(funding_diff) < min_threshold:
                    continue
                
                # Compute the signal
                signal = abs(funding_diff) * open_interest
                signal_ratio = signal / median_signal

                # Add to the list of opportunities
                opportunities.append({
                    'pair': pair,
                    'long_exchange': ex1 if funding_diff < 0 else ex2,  # Shorts pay longs when F < 0
                    'short_exchange': ex1 if funding_diff > 0 else ex2, # Longs pay shorts when F > 0
                    'funding_diff': funding_diff,
                    'abs_funding_diff': abs(funding_diff),
                    'break_even': break_even,
                    'open_interest': open_interest,
                    'signal': signal,
                    'signal_ratio': signal_ratio,
                    'volatility': volatility,
                    'vol_regime': vol_regime
                })

        # Sort opportunities by the strength of the signal ratio
        return sorted(opportunities, key=lambda x: x['signal_ratio'], reverse=True)
        
    def allocate_and_execute(self, opportunities):
        """Allocate capital amongst the best opportunities"""
        if not opportunities:
            return
        
        # Get the available capital
        available_capital = self.calculate_available_capital()

        # Determine how many opportunites to take
        max_opportunities = min(5, len(opportunities))
        selected_opportunities = opportunities[:max_opportunities]

        # For now we simply weigh them equally, but we should weigh based on confidence
        alloc_per_opportunity = available_capital / len(selected_opportunities)

        # Cap allocation
        max_allocation = self.account_equity * self.config['max_position_pct']

        # Execute each trade
        for opp in selected_opportunities:
            base_size = min(alloc_per_opportunity, max_allocation) 

            # Adjust by signal strength, volatiltiy and fee tier
            vol_multiplier = self.get_volatility_multiplier(opp['vol_regime'])
            fee_multiplier = self.calculate_fee_multiplier(self.get_fee_tier(opp['pair']))

            # Compute the position size
            position_size = base_size * opp['signal_ratio'] * vol_multiplier * fee_multiplier
            
            # Execute the trade
            self.execute_trade(
                pair=opp['pair'],
                long_exchange=opp['long_exchange'],
                short_exchange=opp['short_exchange'],
                position_size=position_size,
                funding_diff=opp['funding_diff']
            )


    def calculate_available_capital(self):
        """
        Calculate capital available for new positions with proper risk management.
        This way we keep enough in case of emergency margin calls.
        """
        
        # Maximum portfolio allocation: 75% of total equity
        max_portfolio_allocation = self.account_equity * 0.75
        
        # Calculate capital already deployed in existing positions
        deployed_capital = sum(position['size'] for position in self.active_positions.values())
        
        # Consider margin requirements
        margin_requirements = self.calculate_margin_requirements() * self.config['margin_buffer']
        
        # Ensure we keep enough in reserve for existing positions
        required_reserve = margin_requirements + self.account_equity * 0.10  # 10% additional safety buffer
        
        # Available capital is what remains after accounting for all constraints
        available = max(0, min(
            max_portfolio_allocation - deployed_capital,  # Cap on portfolio allocation
            self.account_equity - required_reserve        # Reserve requirement
        ))
        
        return available
    
    def execute_trade(self, pair, long_ex, short_ex, position_size, funding_diff):
        """Execute the trade by making on the less liquid exchange and taking on the more liquid one"""
        # Analyze orderbooks to determine which exchange has higher impact
        long_impact = self.analyze_orderbook_impact(long_ex, pair, position_size)
        short_impact = self.analyze_orderbook_impact(short_ex, pair, position_size)

        # Determine make/take roles based on liquidity impact
        if long_impact > short_impact:
            # Long exchange has less liquidity, so we make there
            make_ex = long_ex
            take_ex = short_ex
            make_side = "buy"   # Buy on long exchange
            take_side = "sell"  # Sell on short exchange
        else:
            # Short exchange has less liquidity, so we make there
            make_ex = short_ex
            take_ex = long_ex
            make_side = "sell"  # Sell on short exchange
            take_side = "buy"   # Buy on long exchange

        # Get current price for the pair on the make exchange
        current_price = self.get_current_price(make_ex, pair)
        position_units = position_size / current_price

        # Determine optimal chunk size based on the LOB
        chunks = self.calculate_optimal_chunks(
            pair=pair,
            make_exchange=make_ex,
            take_exchange=take_ex,
            position_units=position_units
        )

        executed_units = 0

        # Execute each chunk sequentially for minimal taker hedging impact
        for chunk in chunks:
            # Limit order on taker exchange
            limit_order = self.place_limit_order(
                exchange=make_ex,
                pair=pair,
                side=make_side,
                size=chunk,
                price=self.get_optimal_limit_price(make_ex, pair, make_side)
            )
        
            # Wait for fill with timeout
            filled = self.wait_for_fill(limit_order, timeout=self.config['execution_timeout'] * 60)

            if filled:
                # Immediately hedge with market order on taker exchange
                market_order = self.place_market_order(
                    exchange=take_ex,
                    pair=pair,
                    side=take_side,
                    size=chunk
                )
                
                if market_order['status'] == 'filled':
                    executed_units += chunk
                else:
                    # Handle market order failure
                    self.handle_hedge_failure(make_ex, pair, chunk, make_side)
            else:
                # Adjust limit price if not filled within timeout
                self.adjust_unfilled_order(limit_order)
        
        # Record the position if any units were executed
        if executed_units > 0:
            executed_size = executed_units * current_price
            self.active_positions[pair] = {
                'entry_time': self.current_time(),
                'long_exchange': long_ex,
                'short_exchange': short_ex,
                'make_exchange': make_ex,
                'take_exchange': take_ex,
                'size': executed_size,
                'funding_diff': funding_diff,
                'expected_exit': self.next_funding_after(self.current_time())
            }

    def next_funding_after(self, time):
        """Get the next funding event after given time, with safety buffer for settlement delays"""
        # Get next funding time
        next_funding = self.calculate_next_funding_time(time)
        
        # Add buffer for settlement processing, 5 minutes to be safe
        buffer_minutes = timedelta(minutes=5)
        next_funding_with_buffer = next_funding + buffer_minutes
        
        return next_funding_with_buffer

    def analyze_orderbook_impact(self, exchange, pair, position_size):
        """
        Analyze the orderbook to determine the price impact of a trade
        Returns a measure of impact, where higher ==> less liquidity
        """
        # Get current price
        current_price = self.get_current_price(exchange, pair)
        position_units = position_size / current_price
        
        # Get orderbook
        orderbook = self.get_orderbook(exchange, pair)
        
        # Calculate impact for both sides
        buy_impact = self.calculate_price_impact(orderbook, position_units, "buy")
        sell_impact = self.calculate_price_impact(orderbook, position_units, "sell")
        
        # Return the average impact
        return (buy_impact + sell_impact) / 2
    
    def compute_break_even(self, exchange1, exchange2, pair):
        """
        Calculate the break-even funding differential needed to cover costs
        Assuming best fee tiers on both exchanges
        """
        # Use best exchange fees
        # For example, if taking on Binance and making on OKX
        taker_fee = 0.00017  # Binance best taker fee
        maker_fee = -0.00005 # OKX best maker fee
        
        # Get estimated slippage
        ex1_slippage = self.get_estimated_slippage(exchange1, pair)
        ex2_slippage = self.get_estimated_slippage(exchange2, pair)
        
        # Total one-way cost
        # We always make on exchange with lowest liquidity
        # For all exchanges, a simplified estimate
        total_fees = taker_fee + maker_fee + ex1_slippage + ex2_slippage
        
        # Break-even funding differential, 2x for round trip
        break_even = 2 * total_fees
        
        return break_even
    
    def manage_risk(self):
        """Monitor and manage risk for all active positions"""
        for pair, position in list(self.active_positions.items()):
            # Check if funding event has passed
            current_time = self.current_time()
            if current_time >= position['expected_exit']:
                # We have captured the funding payment
                # Therefore, we close the position
                self.close_position(pair)
                continue
            
            # Check margin usage
            margin_usage = self.get_margin_usage(pair)
            
            # Apply safety triggers if needed
            if margin_usage >= self.config['buffer_triggers']['critical']:
                # Critical level - close position immediately
                self.close_position(pair)
            elif margin_usage >= self.config['buffer_triggers']['danger']:
                # Danger level - reduce by 50%
                self.reduce_position(pair, 0.5)
            elif margin_usage >= self.config['buffer_triggers']['warning']:
                # Warning level - reduce by 20%
                self.reduce_position(pair, 0.2)

    def close_position(self, pair):
        """Close an entire position"""
        if pair not in self.active_positions:
            return
            
        position = self.active_positions[pair]
        
        size = position['size']
        
        # Close both legs of the trade
        current_price = self.get_current_price(position['make_exchange'], pair)
        units = size / current_price
        
        # For closing, we keep the same exchange roles as entry
        make_ex = position['make_exchange']
        take_ex = position['take_exchange']
        
        # Reverse the sides for closing
        make_side = "sell" if position['make_exchange'] == position['long_exchange'] else "buy"
        take_side = "buy" if position['take_exchange'] == position['short_exchange'] else "sell"
        
        self.execute_close_orders(
            pair=pair,
            make_exchange=make_ex,
            take_exchange=take_ex,
            make_side=make_side,
            take_side=take_side,
            units=units
        )
        
        # Remove from active positions
        del self.active_positions[pair]

    def reduce_position(self, pair, reduction_pct):
        """Reduce a position by specified percentage"""
        position = self.active_positions[pair]
        
        # Calculate size to reduce
        current_size = position['size']
        reduction_size = current_size * reduction_pct
        
        # Calculate amount of units to reduce
        current_price = self.get_current_price(position['make_exchange'], pair)
        reduction_units = reduction_size / current_price
        
        # Perform reduction
        make_ex = position['make_exchange']
        take_ex = position['take_exchange']
        make_side = "sell" if position['make_exchange'] == position['long_exchange'] else "buy"
        take_side = "buy" if position['take_exchange'] == position['short_exchange'] else "sell"
        
        # Execute the reduction orders
        self.execute_close_orders(
            pair=pair,
            make_exchange=make_ex,
            take_exchange=take_ex,
            make_side=make_side,
            take_side=take_side,
            units=reduction_units
        )
        
        # Update position size
        position['size'] -= reduction_size

    def calculate_optimal_chunks(self, pair, make_exchange, take_exchange, position_units):
        """Calculate optimal order chunks based on orderbook analysis"""
        # Get orderbook data
        make_orderbook = self.get_orderbook(make_exchange, pair)
        take_orderbook = self.get_orderbook(take_exchange, pair)
        
        # Test different sizes for slippage
        test_sizes = [1, 2, 5, 10, 15, 20, 30]
        optimal_size = test_sizes[0]  # Default to smallest size
        
        for size in test_sizes:
            if size > position_units:
                break
                
            # Calculate slippage for this size
            make_slippage = self.calculate_price_impact(make_orderbook, size, "maker")
            take_slippage = self.calculate_price_impact(take_orderbook, size, "taker")
            
            # Use the worse of the two
            max_slippage = max(make_slippage, take_slippage)
            
            # If exceeds our threshold, use previous size
            if max_slippage > self.config['max_slippage_bps']:
                break
                
            optimal_size = size
        
        # Break position into chunks of optimal size
        chunks = []
        remaining = position_units
        
        while remaining > 0:
            if remaining > optimal_size:
                chunks.append(optimal_size)
                remaining -= optimal_size
            else:
                chunks.append(remaining)
                remaining = 0
                
        return chunks
    
    def calibrate_parameters(self):
        """Calibrate the strategy parameters"""
        current_time = self.current_time()

        # Signal normalization
        if self.days_since_calibration('signal') >= self.config['calibration']['signal_recalibrate']:
            self.calibrate_signal(self.config['calibration']['signal_window'])
            self.last_calibration['signal'] = current_time

        # Volatility thresholds
        if self.days_since_calibration('volatility') >= self.config['calibration']['volatility_recalibrate']:
            self.calibrate_volatility(self.config['calibration']['volatility_window'])
            self.last_calibration['volatility'] = current_time

        # Break-even thresholds
        if self.days_since_calibration('breakeven') >= self.config['calibration']['breakeven_recalibrate']:
            self.calibrate_break_even(self.config['calibration']['breakeven_window'])
            self.last_calibration['breakeven'] = current_time

    
    def get_exchange_combinations(self):
        """
        Generate all unique exchange pairs for comparison.
        """
        # Get list of active exchanges from config
        active_exchanges = []
        
        # Filter to include only exchanges that are currently available
        for exchange in self.config['exchanges']:
            # Skip exchanges that are marked as unavailable
            if self.is_exchange_available(exchange):
                active_exchanges.append(exchange)
        
        # Generate all unique pairs of exchanges
        combinations = []
        for i in range(len(active_exchanges)):
            for j in range(i + 1, len(active_exchanges)):
                ex1 = active_exchanges[i]
                ex2 = active_exchanges[j]                    
                combinations.append((ex1, ex2))
        
        return combinations