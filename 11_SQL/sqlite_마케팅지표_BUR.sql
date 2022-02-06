-- BUR, ARPU, ARPPU
-- ??? ARPDAU(Average Revenue Per DAU) - 유니크 일간 접속 유저 당 결제액


-- buy users 수
SELECT 
	date(payment_date),
	count(distinct customer_id) as dau,
	sum(cast(amount as bigint)) as chang_amount,
	count(CASE
			when amount > 0 then customer_id else null
			end) as buy_users -- 날짜별 구매한 유저 수(중복제거 안함)
from
	payment
group by
	date(payment_date)
order BY 
	date(payment_date);


-- BUR(Buying User Rate)
SELECT
	date(payment_date),
	count(distinct customer_id) as dau,
	sum(cast(amount as bigint)) as chang_amount,
	(count(case when amount > 0 then customer_id else null end) * 1.0 / count(distinct customer_id)) as bur
from
	payment
group by
	date(payment_date)
order BY 
	date(payment_date);


-- ARPU(Average Revenue Per User)-유저 당 평균 결제 금액
SELECT 
	date(payment_date),
	count(distinct customer_id) as dau,
	sum(cast(amount as bigint)) as chang_amount,
	(sum(cast(amount as bigint)) * 1.0) / count(distinct customer_id) as arpu -- 날짜별 구매량/중복제거한 고객수
from
	payment p 
group by
	date(payment_date)
order BY 
	date(payment_date);


-- ARPPU(Average Revenue Per Paying User)-결제 유저 당 평균 결제 금액
SELECT
	date(payment_date),
	count(distinct customer_id) as dau,
	sum(cast(amount as bigint)) as chang_amount,
	(sum(cast(amount as bigint)) * 1.0) / count(case when amount > 0 then customer_id else null end) as arppu
from
	payment p 
group by
	date(payment_date)
order BY 
	date(payment_date);


--  통합
SELECT 
	date(payment_date),
	count(distinct customer_id) as dau,
	sum(cast(amount as bigint)) as chang_amount,
-- buying user 수
	count(CASE
			when amount > 0 then customer_id else null
			end) as buy_users -- 날짜별 구매한 유저 수(중복제거 안함)



SELECT 
	date(payment_date,)
	new