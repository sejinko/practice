-- RFM ? 가치있는 고객을 추출해내어 이를 기준으로 고객을 분류할 수 있는
--       매우 간단하면서도 유용하게 사용될 수 있는 방법으로 알려져 있어서 마케팅에서 자주 사용됨
-- Recency - 거래의 최근성 : 고객이 얼마나 최근에 구입했는가?
-- Frequency - 거래빈도 : 고객이 얼마나 빈번하게 우리 상품을 구입했나?
-- Monetary - 거래규모 : 고객이 구입했던 총 금액은 어느 정도인가?

-- user id, r,f,m 컬럼 추출(전체로 함)
-- r을 구할 때는 평균 하나, 가장 최신 것 하나 이렇게 두 가지를 함


-- r 평균으로 구하기
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






-- lag 컬럼을 구할때 +1을 해버리면 customer_id가 변할때 값이 바뀌어 -가 되버리는 경우도 생김으로 그룹별 +1을 해야 함
-- customer_id last_value를 구해야 함 customer_id의 첫번째가 null값이 나와야 함







-- r 최신것으로 구하기
-- self join 구문
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









-- r,m,f 구하기
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