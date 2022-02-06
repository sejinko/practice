-- DAU(Daily Active User)
-- 신규유져
select 
	date(payment_date),
	customer_id,
	count(payment_id) over(partition by customer_id order by payment_id) as payment_count,
	amount,
	case
		when lag(date(payment_date)) over(partition by customer_id order by date(payment_date)) is null 
		then 1 else 0
	end as new_YN
from 
	payment;


-- 복귀유저, 14는 이탈일이라 정한 일자임
select 
	date(payment_date),
	customer_id,
	count(payment_id) over(partition by customer_id order by payment_id) as payment_count,
	amount,
	case 
		when (strftime('%j',date(payment_date)) - strftime('%j',lag(date(payment_date)) over(partition by customer_id order by date(payment_date)))) > 14 
		then 1 else 0
	end as comeback_YN
from 
	payment;
	

-- 복귀할때까지의 이탈일
select 
	date(payment_date),
	customer_id,
	count(payment_id) over(partition by customer_id order by payment_id) as payment_count,
	amount,
	strftime('%j',date(payment_date)) - strftime('%j',lag(date(payment_date)) over(partition by customer_id order by date(payment_date))) as comeback_Day
from 
	payment;


-- 이탈유저
select 
	date(payment_date),
	customer_id,
	count(payment_id) over(partition by customer_id order by payment_id) as payment_count,
	amount,
	case
		when  
			(ifnull(strftime('%j', lead(date(payment_date)) over(partition by customer_id order by date(payment_date))), max(payment_date) over())) > 14
			- strftime('%j',date(payment_date))
		then 1 else 0
	end out_YN
from
	payment p;
	
	
-- 통합 및 조건걸어서 검색
SELECT 
	*
into tm1
from
	payment;
	



with tm1 as
(
select 
	date(payment_date),
	customer_id,
	count(payment_id) over(partition by customer_id order by payment_id) as payment_count,
	amount,
	-- 신규유저
	case
		when lag(date(payment_date)) over(partition by customer_id order by date(payment_date)) is null 
		then 1 else 0
	end as new_YN,
	-- 복귀유저
	case 
		when (strftime('%j',date(payment_date)) - strftime('%j',lag(date(payment_date)) over(partition by customer_id order by date(payment_date)))) > 14 
		then 1 else 0
	end as comeback_YN,
	-- 복귀까지 이탈 일
	strftime('%j',date(payment_date)) - strftime('%j',lag(date(payment_date)) over(partition by customer_id order by date(payment_date))) as comeback_Day,
	-- 이탈유저
	case
		when  
			(ifnull(strftime('%j', lead(date(payment_date)) over(partition by customer_id order by date(payment_date))), max(payment_date) over())) > 14
			- strftime('%j',date(payment_date))
		then 1 else 0
	end out_YN
FROM 
	payment
)
SELECT 
 *
FROM 
	tm1
where 1=1
and tm1.out_YN = 1
	

-- 그룹바이
with tm1 as
(
select 
	date(payment_date) as new_payment_date,
	customer_id,
	count(payment_id) over(partition by customer_id order by payment_id) as payment_count,
	amount,
	-- 신규유저
	case
		when lag(date(payment_date)) over(partition by customer_id order by date(payment_date)) is null 
		then 1 else 0
	end as new_YN,
	-- 복귀유저
	case 
		when (strftime('%j',date(payment_date)) - strftime('%j',lag(date(payment_date)) over(partition by customer_id order by date(payment_date)))) > 14 
		then 1 else 0
	end as comeback_YN,
	-- 복귀까지 이탈 일
	strftime('%j',date(payment_date)) - strftime('%j',lag(date(payment_date)) over(partition by customer_id order by date(payment_date))) as comeback_Day,
	-- 이탈유저
	case
		when  
			(ifnull(strftime('%j', lead(date(payment_date)) over(partition by customer_id order by date(payment_date))), max(payment_date) over())) > 14
			- strftime('%j',date(payment_date))
		then 1 else 0
	end out_YN
FROM 
	payment
)
SELECT 
	new_payment_date,
	new_YN,
	comeback_YN,
	out_YN,
	count(distinct customer_id) daily_active_users
FROM 
	tm1
group by
	new_payment_date,
	new_YN,
	comeback_YN,
	out_YN;


-- WAU(Weekly Active User)

-- MAU(Mounthly Active User)