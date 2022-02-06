-- DAU
select * from payment limit 10;

select 
	date(payment_date),
	count(distinct customer_id) as daily_active_users
from
	payment p 
group by
	date(payment_date);
	

-- WAU
-- MAU



-- DAU - 신규, 복귀, 이탈 계산
select * from payment limit 10;

  -- 첫번째

select 
	date(payment_date),
	customer_id,
	count(payment_id) over(partition by customer_id order by payment_id),
	amount,
	-- 신규유져
	case
		when lag(date(payment_date)) over(partition by customer_id order by date(payment_date)) is null 
		then 1 else 0
	end as new_YN
from 
	payment;

select 
	date(payment_date),
	customer_id,
	count(payment_id) over(partition by customer_id order by payment_id),
	amount,
	-- 신규유져
	case
		when lag(date(payment_date)) over(partition by customer_id order by date(payment_date)) is null 
		then 1 else 0
	end as new_YN,
	-- 복귀유저, 14는 이탈일이라 정한 일자임
	case 
		when datediff(date(now()), lag(date(payment_date))) over(partition by customer_id order by date(payment_date)) > 14 
		then 1 else 0
	end as comeback_YN,
	datediff(dd, date(now()), lag(date(payment_date))) over(partition by customer_id order by date(payment_date)) as comeback_Day,
	case
		when datediff(dd, date(now()), date(payment_date), isnull(lead(date(payment_date)))) over(partition by account_id order by date(payment_date), '2018-12-31') > 14 
		then 1 else 0 
	end as out_YN
from 
	payment;


  -- 두번째

select 
	date(payment_date),
	customer_id,
	count(payment_id) over(partition by customer_id order by payment_id),
	amount,
	-- 신규유져
	case
		when lag(date(payment_date)) over(partition by customer_id order by date(payment_date)) is null 
		then 1 else 0
	end as new_YN,
	-- 복귀유저, 14는 이탈일이라 정한 일자임
	case 
		when date_part('day', date(now()), lag(date(payment_date))) over(partition by customer_id order by date(payment_date)) > 14 
		then 1 else 0
	end as comeback_YN,
	datediff(dd, date(now()), lag(date(payment_date))) over(partition by customer_id order by date(payment_date)) as comeback_Day,
	case
		when datediff(dd, date(now()), date(payment_date), isnull(lead(date(payment_date)))) over(partition by account_id order by date(payment_date), '2018-12-31') > 14 
		then 1 else 0 
	end as out_YN
from 
	payment;

--- date(now())가 아니라 date(payment_date)가 나와야 하는거 아닌가?



	