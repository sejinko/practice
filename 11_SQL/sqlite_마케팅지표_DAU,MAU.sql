-- DAU(Daily Active User)
-- �ű�����
select 
	date(payment_date),
	customer_id,
	count(payment_id) over(partition by customer_id order by payment_id),
	amount,
	case
		when lag(date(payment_date)) over(partition by customer_id order by date(payment_date)) is null 
		then 1 else 0
	end as new_YN
from 
	payment;


-- ��������, 14�� ��Ż���̶� ���� ������
select 
	date(payment_date),
	customer_id,
	count(payment_id) over(partition by customer_id order by payment_id),
	amount,
	case 
		when (strftime('%j',date(payment_date)) - strftime('%j',lag(date(payment_date)) over(partition by customer_id order by date(payment_date)))) > 14 
		then 1 else 0
	end as comeback_YN
from 
	payment;
	

-- �����Ҷ������� ��Ż��
select 
	date(payment_date),
	customer_id,
	count(payment_id) over(partition by customer_id order by payment_id),
	amount,
	strftime('%j',date(payment_date)) - strftime('%j',lag(date(payment_date)) over(partition by customer_id order by date(payment_date))) as comeback_Day
from 
	payment;


-- 
-- 

-- WAU(Weekly Active User)

-- MAU(Mounthly Active User)