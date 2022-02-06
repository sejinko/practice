-- RFM ? 가치있는 고객을 추출해내어 이를 기준으로 고객을 분류할 수 있는
--       매우 간단하면서도 유용하게 사용될 수 있는 방법으로 알려져 있어서 마케팅에서 자주 사용됨
-- Recency - 거래의 최근성 : 고객이 얼마나 최근에 구입했는가?
-- Frequency - 거래빈도 : 고객이 얼마나 빈번하게 우리 상품을 구입했나?
-- Monetary - 거래규모 : 고객이 구입했던 총 금액은 어느 정도인가?

-- user id, r,f,m 컬럼 추출
-- 월별로 해야하나? 전체로는 못하는 건가?


-- 그럼 r은 구할때 평균으로 구해야 하는 건가?
-- 현재 
select customer_id, AVG(gap)
from
(
	SELECT 
		customer_id,
	--	date(payment_date),
	--	JULIANDAY(date(payment_date)),
	--	lag(date(payment_date)) over(),
		floor(julianday(payment_date)- julianday(lag(date(payment_date)) over())) as gap
	from
		payment
	order by customer_id, payment_date 
)
group by customer_id;

SELECT 
	customer_id,
--	date(payment_date),
--	JULIANDAY(date(payment_date)),
--	lag(date(payment_date)) over(),
	floor(julianday(payment_date)- julianday(lag(date(payment_date)) over())) as gap
from
	payment
order by customer_id, payment_date ;

group by customer_id;

SELECT p1.*, p2.payment_date 
FROM payment p1
left JOIN payment p2
ON p1.payment_id = p2.payment_id+1;
order by customer_id, payment_date;

select ROW_NUMBER() over(), *
from payment;

SELECT 
	date(payment_date),
	customer_id,
	strftime('%j',date(payment_date)) as strf,
	STRFTIME('%j', lag(date(payment_date)) over(order by payment_date)) as lag_strf,
	strftime('%j',date(payment_date))
	- STRFTIME('%j', lag(date(payment_date)) over(order by payment_date)) diff_strf
from
	payment
order by
	payment_date;


SELECT 
	customer_id,
	strftime('%j',date(payment_date)) as strf,
	STRFTIME('%j', lag(date(payment_date)) over()) as lag_strf,
	strftime('%j',date(payment_date)) - STRFTIME('%j', lag(date(payment_date)) over()) as r,
	count(payment_id) as f,
	sum(amount) as m
FROM 
	payment
group by
	customer_id;



-- 분포 구하기
with tm1 as
(
SELECT 
	customer_id,
	strftime('%j',date(payment_date)) as r,
	count(payment_id) as f,
	sum(amount) as m
FROM 
	payment
group by
	customer_id
)
SELECT DISTINCT  
       PERCENTILE_DISC(0.25) WITHIN GROUP (ORDER BY r) OVER (PARTITION BY NULL) as r_25,
       PERCENTILE_DISC(0.5) WITHIN GROUP (ORDER BY r) OVER (PARTITION BY NULL) as r_50,
	   PERCENTILE_DISC(0.75) WITHIN GROUP (ORDER BY r) OVER (PARTITION BY NULL) as r_75,
	   PERCENTILE_DISC(1) WITHIN GROUP (ORDER BY r) OVER (PARTITION BY NULL) as r_100,	   
	   PERCENTILE_DISC(0.25) WITHIN GROUP (ORDER BY f) OVER (PARTITION BY NULL) as f_25,
       PERCENTILE_DISC(0.5) WITHIN GROUP (ORDER BY f) OVER (PARTITION BY NULL) as f_50,
	   PERCENTILE_DISC(0.75) WITHIN GROUP (ORDER BY f) OVER (PARTITION BY NULL) as f_75,
	   PERCENTILE_DISC(1) WITHIN GROUP (ORDER BY f) OVER (PARTITION BY NULL) as f_100,
	   PERCENTILE_DISC(0.25) WITHIN GROUP (ORDER BY m) OVER (PARTITION BY NULL) as m_25,
       PERCENTILE_DISC(0.5) WITHIN GROUP (ORDER BY m) OVER (PARTITION BY NULL) as m_50,
	   PERCENTILE_DISC(0.75) WITHIN GROUP (ORDER BY m) OVER (PARTITION BY NULL) as m_75,
	   PERCENTILE_DISC(1) WITHIN GROUP (ORDER BY m) OVER (PARTITION BY NULL) as m_100
FROM tm1;





-- RFM Score 구하기
SELECT t.*, t.f_score + t.m_score + t.r_score rfm_score
into #rfm
FROM(
	SELECT a.*, case when a.r <= b.r_25 then 1 
					 when a.r <= b.r_50 then 2
					 when a.r <= b.r_75 then 3
					 when a.r <= b.r_100 then 4 else 0 end r_score
			  , case when a.f <= b.f_25 then 1 
					 when a.f <= b.f_50 then 2
					 when a.f <= b.f_75 then 3
					 when a.f <= b.f_100 then 4 else 0 end f_score
			  , case when a.m <= b.m_25 then 1 
					 when a.m <= b.m_50 then 2
					 when a.m <= b.m_75 then 3
					 when a.m <= b.m_100 then 4 else 0 end m_score
	FROM [da_db].[dbo].[RFM] a
Cross join #PERCENTILE b
) as t

select r_score, f_score, m_score, rfm_score, count(distinct userid) [user]
from #rfm
group by r_score, f_score, m_score, rfm_score