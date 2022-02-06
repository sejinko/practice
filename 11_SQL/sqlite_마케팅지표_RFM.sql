-- RFM ? ��ġ�ִ� ���� �����س��� �̸� �������� ���� �з��� �� �ִ�
--       �ſ� �����ϸ鼭�� �����ϰ� ���� �� �ִ� ������� �˷��� �־ �����ÿ��� ���� ����
-- Recency - �ŷ��� �ֱټ� : ���� �󸶳� �ֱٿ� �����ߴ°�?
-- Frequency - �ŷ��� : ���� �󸶳� ����ϰ� �츮 ��ǰ�� �����߳�?
-- Monetary - �ŷ��Ը� : ���� �����ߴ� �� �ݾ��� ��� �����ΰ�?

-- user id, r,f,m �÷� ����
-- ������ �ؾ��ϳ�? ��ü�δ� ���ϴ� �ǰ�?


-- �׷� r�� ���Ҷ� ������� ���ؾ� �ϴ� �ǰ�?
-- ���� 
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



-- ���� ���ϱ�
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





-- RFM Score ���ϱ�
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