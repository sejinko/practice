-- �ǽ� ���� 1�� ��
-- 1����
select
	distinct customer_id,
	amount
from
	payment	
order by amount desc, customer_id asc; 

select
	distinct customer_id,
	amount
from
	payment
where amount = 11.99
order by amount desc, customer_id asc; 

-- 2��°
select top 10
	amount
from
	payment	
order by amount desc, customer_id asc; 

select
	distinct customer_id,
	amount
from
	payment
where amount = 
	(
		select 
			amount 
		from
			payment
		order by amount desc
		limit 1
	)
order by amount desc, customer_id asc; 




-- �ٽ� ����
select
	amount
from
	payment
order by amount desc
limit 1;

select 
	distinct a.customer_id,
	amount
from
	payment a
where 
	a.amount = (
	select 
		amount 
	from
		payment
	order by amount desc 
	limit 1
	)
order by customer_id ;
