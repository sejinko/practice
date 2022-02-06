-- RFM ? ��ġ�ִ� ���� �����س��� �̸� �������� ���� �з��� �� �ִ�
--       �ſ� �����ϸ鼭�� �����ϰ� ���� �� �ִ� ������� �˷��� �־ �����ÿ��� ���� ����
-- Recency - �ŷ��� �ֱټ� : ���� �󸶳� �ֱٿ� �����ߴ°�?
-- Frequency - �ŷ��� : ���� �󸶳� ����ϰ� �츮 ��ǰ�� �����߳�?
-- Monetary - �ŷ��Ը� : ���� �����ߴ� �� �ݾ��� ��� �����ΰ�?

-- user id, r,f,m �÷� ����(��ü�� ��)
-- r�� ���� ���� ��� �ϳ�, ���� �ֽ� �� �ϳ� �̷��� �� ������ ��


-- r ������� ���ϱ�
select
	row_number() over(),
	customer_id,
	AVG(gap)
from
(
	SELECT 
		customer_id,
		floor(julianday(payment_date)- julianday(lag(date(payment_date)) over())) as r
	from
		payment
	order by customer_id, payment_date 
)
group by customer_id;






-- lag �÷��� ���Ҷ� +1�� �ع����� customer_id�� ���Ҷ� ���� �ٲ�� -�� �ǹ����� ��쵵 �������� �׷캰 +1�� �ؾ� ��
-- customer_id last_value�� ���ؾ� �� customer_id�� ù��°�� null���� ���;� ��







-- r �ֽŰ����� ���ϱ�
-- self join ����
select
	customer_id,
	last_payment_date_1,
	last_payment_date_2,
	floor(julianday(last_payment_date_2) - julianday(last_payment_date_1)) as r
from
(
SELECT
	p1.*,
	p2.payment_date as payment_date_2,
	last_value(p1.payment_date) over(partition by p1.customer_id) as last_payment_date_1,
	last_value(p2.payment_date) over(partition by p2.customer_id) as last_payment_date_2
FROM
	payment p1
left JOIN 
	payment p2 ON p1.payment_id = p2.payment_id+1
order by 
	customer_id, payment_date
)
group BY 
	customer_id;




SELECT
	p1.*,
	p2.payment_date as payment_date_2,
	last_value(p1.payment_date) over(partition by p1.customer_id) as last_payment_date_1,
	last_value(p2.payment_date) over(partition by p2.customer_id) as last_payment_date_2
FROM
	payment p1
left JOIN 
	payment p2 ON p1.payment_id = p2.payment_id+1
order by 
	customer_id, payment_date









-- r,m,f ���ϱ�
select
	customer_id,
	last_payment_date_1,
	last_payment_date_2,
	julianday(last_payment_date_2),
	
	julianday(last_payment_date_1),
	floor(julianday(last_payment_date_2) - julianday(last_payment_date_1)) as r,
	count(payment_id) as f,
	sum(amount) as m
from
(
SELECT
	p1.*,
	p2.payment_date as payment_date_2,
	last_value(p1.payment_date) over(partition by p1.customer_id) as last_payment_date_1,
	last_value(p2.payment_date) over(partition by p2.customer_id) as last_payment_date_2
FROM
	payment p1
left JOIN 
	payment p2 ON p1.payment_id = p2.payment_id+1
order by 
	customer_id, payment_date
)
group BY 
	customer_id;
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