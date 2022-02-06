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



-- DAU - �ű�, ����, ��Ż ���
select * from payment limit 10;

  -- ù��°

select 
	date(payment_date),
	customer_id,
	count(payment_id) over(partition by customer_id order by payment_id),
	amount,
	-- �ű�����
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
	-- �ű�����
	case
		when lag(date(payment_date)) over(partition by customer_id order by date(payment_date)) is null 
		then 1 else 0
	end as new_YN,
	-- ��������, 14�� ��Ż���̶� ���� ������
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


  -- �ι�°

select 
	date(payment_date),
	customer_id,
	count(payment_id) over(partition by customer_id order by payment_id),
	amount,
	-- �ű�����
	case
		when lag(date(payment_date)) over(partition by customer_id order by date(payment_date)) is null 
		then 1 else 0
	end as new_YN,
	-- ��������, 14�� ��Ż���̶� ���� ������
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

--- date(now())�� �ƴ϶� date(payment_date)�� ���;� �ϴ°� �ƴѰ�?



	