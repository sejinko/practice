-- BUR, ARPU, ARPPU
-- ??? ARPDAU(Average Revenue Per DAU) - ����ũ �ϰ� ���� ���� �� ������


-- buy users ��
SELECT 
	date(payment_date),
	count(distinct customer_id) as dau,
	count(CASE
			when amount > 0 then customer_id else null
			end) as buy_users -- ��¥�� ������ ���� ��(�ߺ����� ����)
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
	(count(case when amount > 0 then customer_id else null end) * 1.0 / count(distinct customer_id)) as bur
from
	payment
group by
	date(payment_date)
order BY 
	date(payment_date);


-- ARPU(Average Revenue Per User)-���� �� ��� ���� �ݾ�
SELECT 
	date(payment_date),
	count(distinct customer_id) as dau,
	(sum(cast(amount as bigint)) * 1.0) / count(distinct customer_id) as arpu -- ��¥�� ���ŷ�/�ߺ������� ����
from
	payment p 
group by
	date(payment_date)
order BY 
	date(payment_date);


-- ARPPU(Average Revenue Per Paying User)-���� ���� �� ��� ���� �ݾ�
SELECT
	date(payment_date),
	count(distinct customer_id) as dau,
	(sum(cast(amount as bigint)) * 1.0) / count(case when amount > 0 then customer_id else null end) as arppu
from
	payment p 
group by
	date(payment_date)
order BY 
	date(payment_date);


-- ����
SELECT 
	date(payment_date),
	count(distinct customer_id) as dau,
-- buying user ��
	count(CASE when amount > 0 then customer_id else null end) as buy_users, -- ��¥�� ������ ���� ��(�ߺ����� ����)
-- BUR(Buying User Rate)
	(count(case when amount > 0 then customer_id else null end) * 1.0 / count(distinct customer_id)) as bur,
-- ARPU(Average Revenue Per User)-���� �� ��� ���� �ݾ�
	(sum(cast(amount as bigint)) * 1.0) / count(distinct customer_id) as arpu, -- ��¥�� ���ŷ�/�ߺ������� ����
-- ARPPU(Average Revenue Per Paying User)-���� ���� �� ��� ���� �ݾ�
	(sum(cast(amount as bigint)) * 1.0) / count(case when amount > 0 then customer_id else null end) as arppu
from
	payment p
group by
	payment_date 
order BY 
	payment_date;
	

-- �׷����
with tm1 as
(
select 
	date(payment_date) as new_payment_date,
	customer_id,
	count(payment_id) over(partition by customer_id order by payment_id) as payment_count,
	amount,
	-- �ű�����
	case
		when lag(date(payment_date)) over(partition by customer_id order by date(payment_date)) is null 
		then 1 else 0
	end as new_YN,
	-- ��������
	case 
		when (strftime('%j',date(payment_date)) - strftime('%j',lag(date(payment_date)) over(partition by customer_id order by date(payment_date)))) > 14 
		then 1 else 0
	end as comeback_YN,
	-- ���ͱ��� ��Ż ��
	strftime('%j',date(payment_date)) - strftime('%j',lag(date(payment_date)) over(partition by customer_id order by date(payment_date))) as comeback_Day,
	-- ��Ż����
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
	count(distinct customer_id) as dau,
-- buying user ��
	count(CASE when amount > 0 then customer_id else null end) as buy_users, -- ��¥�� ������ ���� ��(�ߺ����� ����)
-- BUR(Buying User Rate)
	(count(case when amount > 0 then customer_id else null end) * 1.0 / count(distinct customer_id)) as bur,
-- ARPU(Average Revenue Per User)-���� �� ��� ���� �ݾ�
	(sum(cast(amount as bigint)) * 1.0) / count(distinct customer_id) as arpu, -- ��¥�� ���ŷ�/�ߺ������� ����
-- ARPPU(Average Revenue Per Paying User)-���� ���� �� ��� ���� �ݾ�
	(sum(cast(amount as bigint)) * 1.0) / count(case when amount > 0 then customer_id else null end) as arppu
from
	tm1
group by
	new_payment_date, new_YN, comeback_YN;
