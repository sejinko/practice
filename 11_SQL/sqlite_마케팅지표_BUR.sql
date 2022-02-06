-- BUR, ARPU, ARPPU
-- ??? ARPDAU(Average Revenue Per DAU) - ����ũ �ϰ� ���� ���� �� ������


-- buy users ��
SELECT 
	date(payment_date),
	count(distinct customer_id) as dau,
	sum(cast(amount as bigint)) as chang_amount,
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
	sum(cast(amount as bigint)) as chang_amount,
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
	sum(cast(amount as bigint)) as chang_amount,
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
	sum(cast(amount as bigint)) as chang_amount,
	(sum(cast(amount as bigint)) * 1.0) / count(case when amount > 0 then customer_id else null end) as arppu
from
	payment p 
group by
	date(payment_date)
order BY 
	date(payment_date);


--  ����
SELECT 
	date(payment_date),
	count(distinct customer_id) as dau,
	sum(cast(amount as bigint)) as chang_amount,
-- buying user ��
	count(CASE
			when amount > 0 then customer_id else null
			end) as buy_users -- ��¥�� ������ ���� ��(�ߺ����� ����)



SELECT 
	date(payment_date,)
	new