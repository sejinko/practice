- coalesce
select * from payment limit 10;

select 
	payment_id,
	amount,
	coalesce(0, rental_id) as not_null_id
from 
	payment;
	

-- 기본 함수
select
	amount,
	abs(amount) as abs_amount,
	ceil(amount) as ceil_amount,
	floor(amount) as floor,
	round(amount,0) as round,
	exp(1) as exp,
	power(3, 2) as power,
	sign(amount) as sign,
	sqrt(amount) as sqrt,
	cast(amount/2 as real)as divide,
	amount/2 as mod_amount
from
	payment;
	

-- group by 함수
select 
	amount,
	min(amount) as min_amount,
	max(amount) as max_amount,
	sum(amount) as sum_amount,
	avg(amount) as avg_amount
from
	payment
group by
	amount;


-- over 함수
select 
	customer_id,
	amount,
	sum(amount) over(partition by customer_id order by amount desc) as sum_amount,
	cume_dist() over(partition by customer_id order by amount desc) as cume_dist_customer_id,
	percent_rank() over(partition by customer_id order by amount desc) as percent_rank_customer_id
from 
	payment;


-- type 변환
-- cast
select 
	'12.54321' as string,
	cast('12.54321' as int),
	cast('12.54321' as integer),
	cast('12.54321' as bigint), 
	cast('12.2' as float),
	cast('12.54321' as double precision),
	cast('12.54321' as numeric),
	cast('12.54321' as numeric(6,2)),
	cast('12.54321' as decimal(6,3)),
	cast('2015-01-01' as date),
	cast('2015-01-01' as timestamp),
	cast(12.54321 as text);
	
	

	