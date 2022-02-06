
/* LTV ���� ���
LTV = ���͡��ŷ� �Ⱓ(������Ÿ��)��������(���� ��ġ ���)
LTV = ���� ���� �ŷ��� �� ���ͷ� �� �� ���� ����
LTV = ���� ��� ���� �ܰ� �� ��� ���� Ƚ��
LTV = (����� - ���� ����) �� ������ ��
LTV = ��� ���� �ܰ� �� ���� �� �� ��� ���� �Ⱓ
LTV = (��� ���� �ܰ� �� ���� �� �� ��� ���� �Ⱓ) -(�ű� ȹ�� ���) + �� ���� ���)
LTV = ARPU / ��Ż��
*/

-- LTV�� �˸� �ű԰� ȹ�� ����� �ľ��� �� ����
-- ���� CPA(��ǥ CPA,Cost Per Action) = LTV X ���� ������

-- LTV(Life Time Value, �� ��� ��ġ)�� ���ϱ� ���� with������ ���� DAU - �ű�, ����, ��Ż ����ϱ�
with tm1 as
(
select 
	date(payment_date) as,
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
-- LTV�� ���ϱ� ���� ������ ����
SELECT
	date(payment_date),
	count(distinct customer_id) dau,
	sum(cast(amount as bigint)) amount,
	count(case when amount > 0 then customer_id else null end) buy_users,
	(sum(cast(amount as bigint)) * 1.0) / count(distinct customer_id) as arpu,
	(count(case when amount > 0 then customer_id else null end) * 1.0) / count(distinct customer_id) as bur,
	(sum(cast(amount as bigint)) * 1.0) / count(case when amount > 0 then customer_id else null end) as arppu,             
	(count(case when out_YN = 1 then customer_id else null end) * 1.0) / count(distinct customer_id) churn_rate
FROM tm1
WHERE 1=1
and new_YN = 1
group by date(payment_date)
order by 1;
